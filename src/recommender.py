import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.similarity_matrix = None

    def build_similarity_matrix(self):
        """Build cosine similarity matrix from food features"""
        features = self.data_processor.get_features()
        self.similarity_matrix = cosine_similarity(features)
        return self.similarity_matrix

    def get_recommendations(self, item_name, healthy_option=False, n_recommendations=5):
        """Get top N similar food items"""
        # Find the index of the input item
        item_idx = self.data_processor.data[self.data_processor.data['item'] == item_name].index[0]

        # Get similarity scores for the item
        item_similarities = self.similarity_matrix[item_idx]

        # Get indices of top similar items (excluding the input item)
        similar_indices = np.argsort(item_similarities)[::-1][1:n_recommendations+1]

        # Get the similar items and their similarity scores
        recommendations = []
        for idx in similar_indices:
            item_data = self.data_processor.data.iloc[idx]

            # If healthy_option is selected, filter items based on calories and protein
            if healthy_option:
                if item_data['calories'] < 500 and item_data['protein'] > 20:
                    recommendations.append({
                        'item': item_data['item'],
                        'restaurant': item_data['restaurant'],
                        'calories': item_data['calories'],
                        'protein': item_data['protein'],
                        'similarity_score': item_similarities[idx]
                    })
            else:
                recommendations.append({
                    'item': item_data['item'],
                    'restaurant': item_data['restaurant'],
                    'calories': item_data['calories'],
                    'protein': item_data['protein'],
                    'similarity_score': item_similarities[idx]
                })

        return recommendations
