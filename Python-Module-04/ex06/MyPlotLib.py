import pandas as pd
import matplotlib.pyplot as plt

class MyPlotLib:
    """
    This class implements different plotting methods for a given pandas.DataFrame.
    Each method takes two arguments:
        - a pandas.DataFrame which contains the dataset,
        - a list of feature names.
    """
    
    def histogram(self, data: pd.DataFrame, features: list) -> None:
        """
        This method plots one histogram for each numerical feature in the list.
        
        Args:
        - data: a pandas.DataFrame which contains the dataset
        - features: a list of feature names
        
        Returns:
        - None
        """
        pass
        
    def density(self, data: pd.DataFrame, features: list) -> None:
        """
        This method plots the density curve of each numerical feature in the list.
        
        Args:
        - data: a pandas.DataFrame which contains the dataset
        - features: a list of feature names
        
        Returns:
        - None
        """
        pass
        
    def pair_plot(self, data: pd.DataFrame, features: list) -> None:
        """
        This method plots a matrix of subplots (also called scatter plot matrix). On each subplot shows a scatter plot of
        one numerical variable against another one. The main diagonal of this matrix shows simple histograms.
        
        Args:
        - data: a pandas.DataFrame which contains the dataset
        - features: a list of feature names
        
        Returns:
        - None
        """
        pass
    
    def box_plot(self, data: pd.DataFrame, features: list) -> None:
        """
        This method displays a box plot for each numerical variable in the dataset.
        
        Args:
        - data: a pandas.DataFrame which contains the dataset
        - features: a list of feature names
        
        Returns:
        - None
        """
        pass
