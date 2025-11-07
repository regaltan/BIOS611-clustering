import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_array, check_random_state
from typing import Union, Callable, Optional, Dict, Any
import warnings

class ClusGap(BaseEstimator):
    """
    Python implementation of R's clusGap function from the cluster package.
    
    Calculates the gap statistic for estimating the optimal number of clusters.
    Based on Tibshirani, Walther, and Hastie (2001).
    
    Parameters
    ----------
    clusterer : object or callable
        Either a sklearn clustering estimator with fit_predict method,
        or a callable function(X, k) that returns cluster labels.
    K_max : int
        Maximum number of clusters to test (must be >= 2).
    B : int, default=100
        Number of bootstrap samples for reference distribution.
        Recommended: 500 for production use.
    d_power : float, default=1
        Power for distance calculation in dispersion.
        1 = sum of distances (R default for historical reasons)
        2 = sum of squared distances (original Tibshirani paper)
    spaceH0 : str, default='scaledPCA'
        Method for generating reference distribution:
        - 'scaledPCA': uniform in PC-aligned box (recommended)
        - 'original': uniform in original space box
    verbose : bool, default=False
        Print progress during computation.
    random_state : int or RandomState, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    gap_values_ : array of shape (K_max,)
        Gap statistic for each k from 1 to K_max.
    s_values_ : array of shape (K_max,)
        Standard error for each k.
    Tab_ : array of shape (K_max, 4)
        Full results table with columns: logW, E.logW, gap, SE.sim
    optimal_k_ : int
        Optimal number of clusters selected.
    n_clusters_ : int
        Same as optimal_k_ (for sklearn compatibility).
    """
    
    def __init__(self, clusterer=None, K_max=10, B=100, d_power=1,
                 spaceH0='scaledPCA', verbose=False, random_state=None):
        self.clusterer = clusterer if clusterer is not None else KMeans()
        self.K_max = K_max
        self.B = B
        self.d_power = d_power
        self.spaceH0 = spaceH0
        self.verbose = verbose
        self.random_state = random_state
        
    def _compute_Wk(self, X, labels, d_power=None):
        """
        Compute within-cluster dispersion W(k).
        
        W_k = 0.5 * sum over clusters r of [sum(distances^d_power) / n_r]
        """
        if d_power is None:
            d_power = self.d_power
            
        unique_labels = np.unique(labels)
        total_dispersion = 0.0
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = X[cluster_mask]
            n_r = len(cluster_points)
            
            if n_r > 1:
                # Compute pairwise distances within cluster
                distances = pdist(cluster_points, metric='euclidean')
                # Sum of powered distances divided by cluster size
                total_dispersion += np.sum(distances ** d_power) / n_r
        
        return 0.5 * total_dispersion
    
    def _cluster_data(self, X, k):
        """
        Cluster data into k clusters using the specified clusterer.
        """
        if k == 1:
            # For k=1, all points in single cluster
            return np.zeros(len(X), dtype=int)
        
        # Handle callable clustering function
        if callable(self.clusterer) and not hasattr(self.clusterer, 'fit_predict'):
            result = self.clusterer(X, k)
            if hasattr(result, 'cluster'):
                return result.cluster
            elif isinstance(result, dict) and 'cluster' in result:
                return result['cluster']
            else:
                return result
        
        # Handle sklearn-style clusterer
        if hasattr(self.clusterer, 'n_clusters'):
            # Clone and set n_clusters
            import copy
            clusterer = copy.deepcopy(self.clusterer)
            clusterer.n_clusters = k
        else:
            clusterer = self.clusterer
            
        if hasattr(clusterer, 'fit_predict'):
            labels = clusterer.fit_predict(X)
        else:
            clusterer.fit(X)
            labels = clusterer.labels_
            
        return labels
    
    def _generate_reference_data(self, X, random_state):
        """
        Generate uniform reference dataset based on spaceH0 method.
        """
        n, p = X.shape
        
        if self.spaceH0 == 'scaledPCA':
            # Center the data
            X_centered = X - np.mean(X, axis=0)
            
            # SVD for PCA transformation
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # Transform to PC space
            X_transformed = X_centered @ Vt.T
            
            # Get ranges in PC space
            mins = X_transformed.min(axis=0)
            maxs = X_transformed.max(axis=0)
            
            # Generate uniform data in PC space
            Z_transformed = random_state.uniform(mins, maxs, size=(n, p))
            
            # Back-transform to original space
            Z = Z_transformed @ Vt + np.mean(X, axis=0)
            
        elif self.spaceH0 == 'original':
            # Center the data
            X_centered = X - np.mean(X, axis=0)
            
            # Get ranges in original space
            mins = X_centered.min(axis=0)
            maxs = X_centered.max(axis=0)
            
            # Generate uniform data in centered space
            Z = random_state.uniform(mins, maxs, size=(n, p)) + np.mean(X, axis=0)
            
        else:
            raise ValueError(f"Invalid spaceH0: {self.spaceH0}")
        
        return Z
    
    def fit(self, X, y=None):
        """
        Calculate gap statistic for clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to cluster.
        y : ignored
            Not used, present for sklearn compatibility.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        random_state = check_random_state(self.random_state)
        
        if self.K_max < 2:
            raise ValueError("K_max must be >= 2")
        
        n, p = X.shape
        if self.K_max > n - 1:
            warnings.warn(f"K_max ({self.K_max}) is large relative to n ({n})")
        
        # Step 1: Cluster observed data for k=1 to K_max
        logW = np.zeros(self.K_max)
        
        if self.verbose:
            print("Clustering observed data...")
            
        for k in range(1, self.K_max + 1):
            labels = self._cluster_data(X, k)
            Wk = self._compute_Wk(X, labels)
            # Handle case where Wk might be 0 (perfect clustering)
            logW[k-1] = np.log(Wk) if Wk > 0 else -np.inf
            
            if self.verbose:
                print(f"  k={k}: W={Wk:.4f}, logW={logW[k-1]:.4f}")
        
        # Step 2: Bootstrap to create reference distribution
        logWks = np.zeros((self.B, self.K_max))
        
        if self.verbose:
            print(f"\nBootstrapping {self.B} reference datasets...")
            
        for b in range(self.B):
            if self.verbose and (b + 1) % 10 == 0:
                print(f"  Bootstrap sample {b+1}/{self.B}")
                
            # Generate reference data
            Z = self._generate_reference_data(X, random_state)
            
            # Cluster reference data for all k
            for k in range(1, self.K_max + 1):
                labels = self._cluster_data(Z, k)
                Wk_ref = self._compute_Wk(Z, labels)
                logWks[b, k-1] = np.log(Wk_ref) if Wk_ref > 0 else -np.inf
        
        # Step 3: Calculate gap statistics
        E_logW = np.mean(logWks, axis=0)
        sd_logW = np.std(logWks, axis=0, ddof=1)
        
        # Standard error with simulation correction factor
        SE_sim = np.sqrt(1 + 1/self.B) * sd_logW
        
        # Gap statistic
        gap = E_logW - logW
        
        # Create results table (matching R's structure)
        self.Tab_ = np.column_stack([logW, E_logW, gap, SE_sim])
        self.gap_values_ = gap
        self.s_values_ = SE_sim
        
        # Determine optimal k using different methods
        self.optimal_k_ = self.select_optimal_k(method='Tibs2001SEmax')
        self.n_clusters_ = self.optimal_k_  # sklearn compatibility
        
        # Store additional metadata
        self.logWks_ = logWks
        self.n_ = n
        self.B_ = self.B
        self.spaceH0_ = self.spaceH0
        
        if self.verbose:
            print(f"\nOptimal number of clusters: {self.optimal_k_}")
        
        return self
    
    def select_optimal_k(self, method='Tibs2001SEmax', SE_factor=1):
        """
        Select optimal number of clusters using various methods.
        
        Parameters
        ----------
        method : str, default='Tibs2001SEmax'
            Method for selecting optimal k:
            - 'Tibs2001SEmax': Original Tibshirani et al. 2001 method
            - 'firstSEmax': Smallest k within SE_factor*SE of first local max
            - 'globalSEmax': k within SE_factor*SE of global maximum
            - 'firstmax': First local maximum (ignores SE)
            - 'globalmax': Global maximum (ignores SE)
        SE_factor : float, default=1
            Multiplier for standard error in selection rules.
            
        Returns
        -------
        optimal_k : int
            Optimal number of clusters (1-indexed).
        """
        gap = self.gap_values_
        s = self.s_values_
        K_max = len(gap)
        
        if method == 'Tibs2001SEmax':
            # Original Tibshirani rule: smallest k where gap[k] >= gap[k+1] - s[k+1]
            for k in range(K_max - 1):
                if gap[k] >= gap[k+1] - SE_factor * s[k+1]:
                    return k + 1  # Convert to 1-indexed
            return K_max
            
        elif method == 'firstSEmax':
            # Find first local maximum
            local_max_idx = None
            for k in range(1, K_max - 1):
                if gap[k] > gap[k-1] and gap[k] > gap[k+1]:
                    local_max_idx = k
                    break
            if local_max_idx is None:
                local_max_idx = np.argmax(gap)
                
            # Find smallest k within SE_factor*SE of this maximum
            threshold = gap[local_max_idx] - SE_factor * s[local_max_idx]
            for k in range(local_max_idx + 1):
                if gap[k] >= threshold:
                    return k + 1
            return local_max_idx + 1
            
        elif method == 'globalSEmax':
            # Global maximum with SE consideration
            max_idx = np.argmax(gap)
            threshold = gap[max_idx] - SE_factor * s[max_idx]
            for k in range(K_max):
                if gap[k] >= threshold:
                    return k + 1
            return max_idx + 1
            
        elif method == 'firstmax':
            # First local maximum (no SE)
            for k in range(1, K_max - 1):
                if gap[k] > gap[k-1] and gap[k] > gap[k+1]:
                    return k + 1
            return np.argmax(gap) + 1
            
        elif method == 'globalmax':
            # Global maximum (no SE)
            return np.argmax(gap) + 1
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def plot(self, SE_factor=1):
        """
        Plot gap statistic with error bars (requires matplotlib).
        
        Parameters
        ----------
        SE_factor : float, default=1
            Multiplier for error bars display.
            
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        if not hasattr(self, 'gap_values_'):
            raise ValueError("Must call fit() before plotting")
        
        k_values = np.arange(1, self.K_max + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Gap statistic with error bars
        ax1.errorbar(k_values, self.gap_values_, 
                    yerr=SE_factor * self.s_values_,
                    marker='o', capsize=5)
        ax1.axvline(self.optimal_k_, color='red', linestyle='--', 
                   label=f'Optimal k={self.optimal_k_}')
        ax1.set_xlabel('Number of clusters k')
        ax1.set_ylabel('Gap statistic')
        ax1.set_title('Gap Statistic')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Log(W_k) for observed and expected
        ax2.plot(k_values, self.Tab_[:, 0], 'o-', label='Observed log(W_k)')
        ax2.plot(k_values, self.Tab_[:, 1], 's-', label='Expected log(W_k)')
        ax2.axvline(self.optimal_k_, color='red', linestyle='--')
        ax2.set_xlabel('Number of clusters k')
        ax2.set_ylabel('log(W_k)')
        ax2.set_title('Within-cluster Dispersion')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig, ax1
    
    def get_results_dataframe(self):
        """
        Get results as pandas DataFrame (matching R's clusGap output).
        
        Returns
        -------
        df : pandas.DataFrame
            Results with columns: k, logW, E.logW, gap, SE.sim
        """
        if not hasattr(self, 'Tab_'):
            raise ValueError("Must call fit() before getting results")
            
        df = pd.DataFrame(
            self.Tab_,
            columns=['logW', 'E.logW', 'gap', 'SE.sim'],
            index=range(1, self.K_max + 1)
        )
        df.index.name = 'k'
        return df


# Convenience function matching R's interface
def clusGap(x, FUNcluster, K_max, B=100, d_power=1, 
            spaceH0='scaledPCA', verbose=False, random_state=None, **kwargs):
    """
    Direct function interface matching R's clusGap.
    
    Parameters
    ----------
    x : array-like
        Data to cluster.
    FUNcluster : callable or estimator
        Clustering function or sklearn estimator.
    K_max : int
        Maximum number of clusters.
    B : int
        Number of bootstrap samples.
    d_power : float
        Power for distance calculation.
    spaceH0 : str
        Reference distribution method.
    verbose : bool
        Print progress.
    random_state : int or RandomState
        Random seed.
    **kwargs : dict
        Additional arguments passed to clustering function.
        
    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'Tab': results table
        - 'gap': gap values
        - 'SE.sim': standard errors
        - 'optimal_k': selected number of clusters
        - 'n': sample size
        - 'B': number of bootstraps
        - 'FUNcluster': clustering function used
    """
    # Handle additional clustering arguments
    if kwargs and hasattr(FUNcluster, 'set_params'):
        FUNcluster.set_params(**kwargs)
    
    # Create and fit ClusGap object
    gap_stat = ClusGap(
        clusterer=FUNcluster,
        K_max=K_max,
        B=B,
        d_power=d_power,
        spaceH0=spaceH0,
        verbose=verbose,
        random_state=random_state
    )
    
    gap_stat.fit(x)
    
    # Return R-style results dictionary
    return {
        'Tab': gap_stat.Tab_,
        'gap': gap_stat.gap_values_,
        'SE.sim': gap_stat.s_values_,
        'optimal_k': gap_stat.optimal_k_,
        'n': gap_stat.n_,
        'B': gap_stat.B_,
        'FUNcluster': FUNcluster,
        'spaceH0': gap_stat.spaceH0_
    }


# Example wrapper functions for different clustering algorithms
def kmeans_wrapper(n_start=25, **kwargs):
    """Create kmeans clustering function with multiple starts."""
    def cluster_fn(X, k):
        km = KMeans(n_clusters=k, n_init=n_start, **kwargs)
        return km.fit_predict(X)
    return cluster_fn


def hierarchical_wrapper(method='ward', metric='euclidean'):
    """Create hierarchical clustering function."""
    from scipy.cluster.hierarchy import linkage, fcluster
    def cluster_fn(X, k):
        Z = linkage(X, method=method, metric=metric)
        return fcluster(Z, k, criterion='maxclust') - 1  # Convert to 0-indexed
    return cluster_fn