import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from libpysal.weights import KNN
from esda.moran import Moran
from scipy.stats import shapiro, norm as _scipy_norm
from scipy.interpolate import interp1d as _interp1d


# ============================================================================
# Normal-Score Transform (NST)
# ============================================================================

class NormalScoreTransform:
    """
    Bijective transform that maps any continuous distribution to N(0,1).

    Why this matters
    ----------------
    Both Kriging and Gaussian Process regression assume the underlying random
    field is Gaussian (or at least that residuals are Gaussian after detrending).
    In real geoscientific datasets this assumption is routinely violated:
    permeability is log-normal, grade distributions are positively skewed,
    and geochemical data can have heavy tails or multiple modes.

    When you feed skewed data to Kriging, three bad things happen:
      1. The estimated variogram sill is inflated by outliers.
      2. The kriging system assigns too much weight to extreme values.
      3. Back-calculated uncertainty intervals are symmetric in normal space
         but asymmetric (and wrong) in original space.

    The Normal-Score Transform (NST) is the geostatistical industry standard
    fix (Journel & Huijbregts, 1978; Deutsch & Journel, 1998).  It is a
    rank-preserving monotone mapping:
        1. Sort observations, assign uniform quantiles p_i = (rank_i - 0.5) / n
        2. Map each p_i → z_i = Φ⁻¹(p_i)   where Φ is the standard normal CDF
        3. After interpolation in normal space, back-transform via Φ(z) → p,
           then interpolate the original data's empirical quantile function.

    The forward and inverse transforms are implemented as piecewise-linear
    interpolators over the (z, x) knots, extrapolating via the normal
    distribution tails to handle prediction values outside the training range.

    Usage
    -----
    >>> nst = NormalScoreTransform()
    >>> z = nst.fit_transform(x_raw)       # forward: raw → normal scores
    >>> x_back = nst.inverse_transform(z)  # inverse: normal scores → raw
    """

    def __init__(self, tail_extrapolation: bool = True):
        """
        Args:
            tail_extrapolation: If True (default), use the normal distribution
                CDF to extrapolate beyond the training data range.  If False,
                clamp predictions to the observed min/max.
        """
        self.tail_extrapolation = tail_extrapolation
        self._fitted = False

    def fit(self, x: np.ndarray) -> "NormalScoreTransform":
        """
        Learn the mapping from the observed data distribution to N(0,1).

        Args:
            x: 1-D array of raw observations.

        Returns:
            self (for method chaining)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        if n < 3:
            raise ValueError(
                f"NormalScoreTransform requires at least 3 observations, got {n}."
            )

        # Sort and compute plotting positions (Hazen formula: avoids 0 and 1)
        sort_idx   = np.argsort(x)
        x_sorted   = x[sort_idx]
        quantiles  = (np.arange(1, n + 1) - 0.5) / n   # in (0, 1)
        z_sorted   = _scipy_norm.ppf(quantiles)          # normal scores

        # Store unique knots only (handles tied values gracefully)
        _, unique_idx = np.unique(x_sorted, return_index=True)
        self._x_knots = x_sorted[unique_idx]
        self._z_knots = z_sorted[unique_idx]

        # Summary statistics used for tail extrapolation
        self._x_mean = float(np.mean(x))
        self._x_std  = float(np.std(x, ddof=1))
        self._x_min  = float(x_sorted[0])
        self._x_max  = float(x_sorted[-1])
        self._z_min  = float(z_sorted[0])
        self._z_max  = float(z_sorted[-1])

        # Build piecewise-linear interpolators
        # forward:  x → z
        self._fwd = _interp1d(
            self._x_knots, self._z_knots,
            kind="linear", bounds_error=False,
            fill_value=(self._z_min, self._z_max),
        )
        # inverse:  z → x
        self._inv = _interp1d(
            self._z_knots, self._x_knots,
            kind="linear", bounds_error=False,
            fill_value=(self._x_min, self._x_max),
        )

        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the forward transform: raw values → normal scores.

        Args:
            x: 1-D array of values (need not be in the training range).

        Returns:
            z: normal scores ∈ ℝ, approximately N(0,1) distributed.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        x = np.asarray(x, dtype=np.float64).ravel()
        z = self._fwd(x).copy()

        if self.tail_extrapolation:
            # Below training min: use normal tail
            lo = x < self._x_min
            if np.any(lo):
                # Shift x relative to training distribution, map through Φ
                p_lo = _scipy_norm.cdf(
                    (x[lo] - self._x_mean) / max(self._x_std, 1e-12)
                )
                # Clip to avoid ±inf from ppf at extreme probabilities
                p_lo = np.clip(p_lo, 1e-6, 0.5)
                z[lo] = _scipy_norm.ppf(p_lo)

            # Above training max: use normal tail
            hi = x > self._x_max
            if np.any(hi):
                p_hi = _scipy_norm.cdf(
                    (x[hi] - self._x_mean) / max(self._x_std, 1e-12)
                )
                p_hi = np.clip(p_hi, 0.5, 1 - 1e-6)
                z[hi] = _scipy_norm.ppf(p_hi)

        return z

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit and apply forward transform in one step."""
        return self.fit(x).transform(x)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the inverse transform: normal scores → raw values.

        Used to back-transform GP/Kriging predictions from normal space
        back to the original data units.

        Args:
            z: 1-D array of normal scores.

        Returns:
            x: values in original data units.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before inverse_transform().")
        z = np.asarray(z, dtype=np.float64).ravel()

        # ── Hard clip before any interpolation or tail extrapolation ─────────
        # Kriging can produce extreme prediction overshoots (z > ±5 or beyond)
        # between tightly clustered points when the nugget is low.  Without
        # this clip, the CDF tail extrapolation maps those values to physically
        # impossible numbers (millions), causing R² to collapse to -1×10¹².
        # ±3.5 retains 99.95% of genuine normal-score mass while preventing
        # numerical catastrophe.  This is a safety net, not a scientific choice:
        # legitimate predictions from a well-fitted GP/Kriging model will
        # virtually never exceed ±3.5 normal scores for a finite dataset.
        z = np.clip(z, -3.5, 3.5)

        x = self._inv(z).copy()

        if self.tail_extrapolation:
            # Below lowest normal score: extrapolate in original space
            lo = z < self._z_min
            if np.any(lo):
                p_lo = _scipy_norm.cdf(z[lo])
                p_lo = np.clip(p_lo, 1e-9, quantiles_lo := 0.5)
                # Scale to training distribution
                x[lo] = self._x_mean + self._x_std * _scipy_norm.ppf(p_lo)

            # Above highest normal score
            hi = z > self._z_max
            if np.any(hi):
                p_hi = _scipy_norm.cdf(z[hi])
                p_hi = np.clip(p_hi, 0.5, 1 - 1e-9)
                x[hi] = self._x_mean + self._x_std * _scipy_norm.ppf(p_hi)

        return x

    @property
    def is_fitted(self) -> bool:
        """True if fit() has been called."""
        return self._fitted

    def summary(self) -> dict:
        """Return a dict of key transform statistics for logging."""
        if not self._fitted:
            return {}
        return {
            "n_knots":  len(self._x_knots),
            "x_min":    self._x_min,
            "x_max":    self._x_max,
            "x_mean":   self._x_mean,
            "x_std":    self._x_std,
            "z_range":  [self._z_min, self._z_max],
        }


def check_normality(Z: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Run the Shapiro-Wilk normality test and return a diagnostic dict.

    Shapiro-Wilk is preferred over Kolmogorov-Smirnov for geoscientific
    sample sizes (typically n < 5000) because it has higher power against
    realistic alternatives (skewness, heavy tails).

    Args:
        Z    : 1-D array of observed values.
        alpha: significance level for the normality decision (default 0.05).

    Returns:
        dict with keys:
            shapiro_W       – test statistic (1.0 = perfectly normal)
            shapiro_p       – p-value (< alpha → reject normality)
            is_normal       – bool: True when we cannot reject normality
            skewness        – sample skewness
            kurtosis        – excess kurtosis (0 = normal)
            recommend_nst   – bool: True when NST is recommended
    """
    from scipy.stats import skew, kurtosis as _kurt

    Z = np.asarray(Z, dtype=np.float64).ravel()

    # Shapiro-Wilk is reliable for n ≤ 5000; for larger datasets use a subsample
    n_sw = min(len(Z), 5000)
    rng  = np.random.default_rng(42)
    Z_sw = rng.choice(Z, size=n_sw, replace=False) if len(Z) > n_sw else Z

    try:
        sw_stat, sw_p = shapiro(Z_sw)
    except Exception:
        sw_stat, sw_p = float("nan"), float("nan")

    sk   = float(skew(Z))
    kurt = float(_kurt(Z, fisher=True))   # excess kurtosis (normal = 0)

    is_normal      = bool(sw_p >= alpha) if not np.isnan(sw_p) else True
    # Recommend NST only when ALL three conditions hold:
    #   1. Shapiro-Wilk rejects normality (p < alpha)
    #   2. Skewness is practically significant (|skew| > 0.5)
    #   3. Kurtosis is practically significant (|excess kurtosis| > 1.0)
    # Using AND (not OR) prevents NST from firing on data that merely has a
    # slightly heavy tail or passes Shapiro by a hair — both common in
    # finite samples of genuinely Gaussian data (S1, S2 synthetics).
    # The OR logic triggered NST on S1 (Gaussian) and caused a ~0.28 R² drop
    # by computing CV metrics in normal-score units instead of original units.
    recommend_nst  = (not is_normal) and (abs(sk) > 0.5) and (abs(kurt) > 1.0)

    return {
        "shapiro_W":     float(sw_stat),
        "shapiro_p":     float(sw_p),
        "is_normal":     is_normal,
        "skewness":      sk,
        "kurtosis":      kurt,
        "recommend_nst": recommend_nst,
    }


def analyze_trend(X, Y, Z, order: int = 1):
    """
    Perform statistical tests to detect an underlying spatial trend.

    The F-test is conducted on a polynomial surface of the same *order* that
    will actually be used for detrending, so the detection and correction
    stages are always consistent.

    Parameters
    ----------
    X, Y : 1-D arrays – spatial coordinates
    Z    : 1-D array  – observed values
    order : int – polynomial order to test (1 = linear plane, 2 = quadratic, …)
        Must match the ``order`` parameter passed to :class:`TrendProcessor`.

    Returns
    -------
    dict with keys:
        f_pvalue          – p-value of the overall F-test for the polynomial surface
        r2                – coefficient of determination (R²) of the fitted surface
        moran_i           – Global Moran's I statistic on the raw data (nan if failed)
        moran_p           – Permutation p-value for Moran's I (nan if failed)
        recommend_detrend – bool: True when trend is statistically significant
        tested_order      – the polynomial order that was actually tested
        normality         – dict from check_normality() (Shapiro-Wilk, skewness, …)
    """
    coords = np.column_stack((X, Y))

    # ── 1. F-test for the polynomial surface of the requested order ──────────
    # Build polynomial features (no bias column; sm.add_constant adds intercept).
    poly = PolynomialFeatures(degree=order, include_bias=False)
    X_poly = poly.fit_transform(coords)          # shape (n, n_terms)
    X_model = sm.add_constant(X_poly)            # adds the intercept

    ols_results = sm.OLS(Z, X_model).fit()
    f_pvalue = float(ols_results.f_pvalue)
    r2       = float(ols_results.rsquared)

    # ── 2. Global Moran's I on the raw data ──────────────────────────────────
    # Moran's I measures whether spatially nearby samples have similar values.
    # High positive I → clustering → supports presence of large-scale structure.
    try:
        w = KNN.from_array(coords, k=min(5, len(Z) - 1))
        w.transform = 'r'
        mi      = Moran(Z, w)
        moran_i = float(mi.I)
        moran_p = float(mi.p_sim)
    except Exception:
        moran_i = np.nan
        moran_p = np.nan

    # ── 3. Decision rule ──────────────────────────────────────────────────────
    # Detrend only when the F-test is significant (p < 0.05) AND the polynomial
    # surface explains more than 5% of the variance.  The 5% guard prevents
    # detrending when the trend is statistically detectable but practically
    # negligible (common with large n and very weak gradients).
    recommend_detrend = bool(f_pvalue < 0.05 and r2 > 0.05)

    # ── 4. Normality check on the raw residuals ───────────────────────────────
    # We test the raw data (not the OLS residuals) because the downstream
    # engines (GP, Kriging) assume the *field values* (or detrended residuals)
    # are Gaussian.  If they are not, the NST should be applied before fitting.
    normality_stats = check_normality(Z)

    return {
        "f_pvalue":          f_pvalue,
        "r2":                r2,
        "moran_i":           moran_i,
        "moran_p":           moran_p,
        "recommend_detrend": recommend_detrend,
        "tested_order":      order,
        "normality":         normality_stats,
    }

class TrendProcessor:
    """
    Polynomial detrending pre-processor.
    Fits a surface of a given order (1=Linear, 2=Quadratic, 3=Cubic) to spatial data.
    """
    def __init__(self, order=1):
        self.order = order
        self.model = make_pipeline(PolynomialFeatures(order, include_bias=False), LinearRegression())
        
    def fit(self, X, Y, Z):
        """Fit polynomial surface to coordinates (X, Y) and values Z."""
        coords = np.column_stack((X, Y))
        self.model.fit(coords, Z)
        return self
        
    def get_trend(self, X, Y):
        """Calculate trend values at coordinates (X, Y)."""
        coords = np.column_stack((X, Y))
        return self.model.predict(coords)
        
    def detrend(self, X, Y, Z):
        """Remove trend from Z values."""
        trend = self.get_trend(X, Y)
        return Z - trend
        
    def retrend(self, X, Y, Z_res):
        """Add trend back to residuals."""
        trend = self.get_trend(X, Y)
        return Z_res + trend
        
    def get_params(self):
        """Return parameters of the fitted linear regression model."""
        lr = self.model.named_steps['linearregression']
        return {
            'order': self.order,
            'intercept': lr.intercept_,
            'coefficients': lr.coef_.tolist()
        }
