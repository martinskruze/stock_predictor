"""
Utility for classifying price differences into categories
"""

class PriceClassifier:
    """
    Classifies price differences into numeric categories
    """

    @staticmethod
    def classify(current_price, reference_price):
        """
        Classify the absolute difference between two prices

        Args:
            current_price (float): The current price
            reference_price (float): The reference price to compare against

        Returns:
            int: A classification type from 1-7 based on price difference

        Classification:
            Type 0: diff > 10
            Type 1: 5 < diff <= 10
            Type 2: 1 < diff <= 5
            Type 3: -1 <= diff <= 1
            Type 4: -5 <= diff < -1
            Type 5: -10 <= diff < -5
            Type 6: diff < -10
        """
        if reference_price is None:
            return 3  # neutral type if no reference

        # Calculate absolute price difference
        price_diff = current_price - reference_price

        # Classify based on absolute price difference
        if price_diff > 10:
            return 0
        elif 5 < price_diff <= 10:
            return 1
        elif 1 < price_diff <= 5:
            return 2
        elif -1 <= price_diff <= 1:
            return 3
        elif -5 <= price_diff < -1:
            return 4
        elif -10 <= price_diff < -5:
            return 5
        else:  # price_diff < -10
            return 6

    @staticmethod
    def description(price_type):
        """
        Get a human-readable description for a price classification type

        Args:
            price_type (int): The classification type (1-7)

        Returns:
            str: Human-readable description of the price classification
        """
        descriptions = {
            0: "Strong rise (more than $10 increase)",
            1: "Moderate rise ($5 to $10 increase)",
            2: "Slight rise ($1 to $5 increase)",
            3: "Stable (within $1 change)",
            4: "Slight drop ($1 to $5 decrease)",
            5: "Moderate drop ($5 to $10 decrease)",
            6: "Strong drop (more than $10 decrease)"
        }

        return descriptions.get(price_type, "Unknown classification type")
