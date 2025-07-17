"""
Model loader utility for recipe image search
"""

import pickle
import numpy as np
import os
import pandas as pd
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

_custom_layers_defined = False
L2NormalizationLayer = None
ExtractTripletComponent = None
TripletStackLayer = None

def _define_custom_layers():
    global _custom_layers_defined, L2NormalizationLayer, ExtractTripletComponent, TripletStackLayer
    
    if _custom_layers_defined:
        return
    
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    
    class _L2NormalizationLayer(Layer):
        
        def __init__(self, axis=1, **kwargs):
            super(_L2NormalizationLayer, self).__init__(**kwargs)
            self.axis = axis
        
        def call(self, inputs):
            return tf.nn.l2_normalize(inputs, axis=self.axis)
        
        def compute_output_shape(self, input_shape):
            return input_shape
        
        def get_config(self):
            config = super(_L2NormalizationLayer, self).get_config()
            config.update({'axis': self.axis})
            return config

    class _ExtractTripletComponent(Layer):
        
        def __init__(self, component_index, **kwargs):
            super(_ExtractTripletComponent, self).__init__(**kwargs)
            self.component_index = component_index
        
        def call(self, inputs):
            return inputs[:, self.component_index]
        
        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[2], input_shape[3], input_shape[4])
        
        def get_config(self):
            config = super(_ExtractTripletComponent, self).get_config()
            config.update({'component_index': self.component_index})
            return config

    class _TripletStackLayer(Layer):
        
        def __init__(self, **kwargs):
            super(_TripletStackLayer, self).__init__(**kwargs)
        
        def call(self, inputs):
            return tf.stack(inputs, axis=1)
        
        def compute_output_shape(self, input_shape):
            batch_size = input_shape[0][0]
            embedding_dim = input_shape[0][1]
            return (batch_size, 3, embedding_dim)
        
        def get_config(self):
            return super(_TripletStackLayer, self).get_config()

    L2NormalizationLayer = _L2NormalizationLayer
    ExtractTripletComponent = _ExtractTripletComponent
    TripletStackLayer = _TripletStackLayer
    
    _custom_layers_defined = True


def _get_custom_loss_and_metrics():
    import tensorflow as tf
    
    # TRIPLET LOSS OPTIMISÉE POUR EMBEDDINGS NORMALISÉS
    def triplet_loss(margin=0.3):
        """Triplet loss optimisée pour embeddings L2-normalisés"""
        def triplet_loss_fn(y_true, y_pred):
            """
            y_pred contient [anchor, positive, negative] embeddings normalisés
            Utilise distance cosinus (1 - similarity) pour la loss
            """
            anchor = y_pred[:, 0, :]      # (batch_size, embedding_dim)
            positive = y_pred[:, 1, :]    # (batch_size, embedding_dim)
            negative = y_pred[:, 2, :]    # (batch_size, embedding_dim)
            
            # Calcul des distances cosinus (1 - similarité cosinus)
            # Pour embeddings normalisés: cosine_sim = dot_product
            pos_similarity = tf.reduce_sum(anchor * positive, axis=1)
            neg_similarity = tf.reduce_sum(anchor * negative, axis=1)
            
            pos_distance = 1.0 - pos_similarity
            neg_distance = 1.0 - neg_similarity
            
            # Triplet loss: max(0, pos_dist - neg_dist + margin)
            basic_loss = pos_distance - neg_distance + margin
            loss = tf.maximum(0.0, basic_loss)
            
            return tf.reduce_mean(loss)
        
        return triplet_loss_fn

    def triplet_margin_accuracy(y_true, y_pred):
        """% de triplets qui respectent la marge complète (vraie réussite)"""
        anchor = y_pred[:, 0, :]
        positive = y_pred[:, 1, :]
        negative = y_pred[:, 2, :]
        
        pos_similarity = tf.reduce_sum(anchor * positive, axis=1)
        neg_similarity = tf.reduce_sum(anchor * negative, axis=1)
        
        # Distance cosinus
        pos_distance = 1.0 - pos_similarity
        neg_distance = 1.0 - neg_similarity
        
        # Triplet satisfait si : neg_dist - pos_dist > margin
        margin = 0.3  # Utiliser la marge par défaut
        margin_satisfied = tf.cast(neg_distance - pos_distance > margin, tf.float32)
        
        return tf.reduce_mean(margin_satisfied)

    def average_positive_similarity(y_true, y_pred):
        """Similarité moyenne anchor-positive"""
        anchor = y_pred[:, 0, :]
        positive = y_pred[:, 1, :]
        pos_similarity = tf.reduce_sum(anchor * positive, axis=1)
        return tf.reduce_mean(pos_similarity)

    def average_negative_similarity(y_true, y_pred):
        """Similarité moyenne anchor-negative"""
        anchor = y_pred[:, 0, :]
        negative = y_pred[:, 2, :]
        neg_similarity = tf.reduce_sum(anchor * negative, axis=1)
        return tf.reduce_mean(neg_similarity)
    
    return {
        'triplet_loss': triplet_loss(),
        'triplet_margin_accuracy': triplet_margin_accuracy,
        'average_positive_similarity': average_positive_similarity,
        'average_negative_similarity': average_negative_similarity
    }


class RecipeImageRetrievalFT:
    """Recipe image search system"""
    
    def __init__(self):
        self.model = None
        self.embeddings_db = None
        self.image_paths = None
        self.image_to_recipe_map = None
        self.recipes_df = None
        self.img_size = 224
        
    def load_system(self, 
                   model_path: str = "./ft/best_embedding_recipe_image_retrieval_model_ft.keras",
                   embeddings_path: str = "./ft/recipe_embeddings_database_ft.npy",
                   metadata_path: str = "./ft/recipe_embeddings_database_metadata_ft.pkl",
                   recipes_df_path: str = "./data/recipes_with_images_dataframe.pkl") -> bool:
        """
        Load the search system with all components
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            logger.info("Loading search system...")
            
            if os.path.exists(embeddings_path):
                self.embeddings_db = np.load(embeddings_path)
                logger.info(f"Embeddings loaded: {self.embeddings_db.shape}")
            else:
                logger.error(f"Embeddings file not found: {embeddings_path}")
                return False
                
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.image_paths = metadata['image_paths']
                self.image_to_recipe_map = metadata['image_to_recipe_map']
                logger.info("Metadata loaded")
            else:
                logger.error(f"Metadata file not found: {metadata_path}")
                return False
                
            if os.path.exists(recipes_df_path):
                self.recipes_df = pd.read_pickle(recipes_df_path)
                logger.info(f"Recipes DataFrame loaded: {len(self.recipes_df)} recipes")
            else:
                logger.warning(f"Recipes DataFrame not found: {recipes_df_path}")
            
            # Load TensorFlow model at startup for production-ready performance
            logger.info("Loading TensorFlow model...")
            import tensorflow as tf
            _define_custom_layers()
            
            custom_objects = {
                'L2NormalizationLayer': L2NormalizationLayer,
                'ExtractTripletComponent': ExtractTripletComponent,
                'TripletStackLayer': TripletStackLayer
            }
            
            loss_and_metrics = _get_custom_loss_and_metrics()
            custom_objects.update(loss_and_metrics)
            
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                logger.info(f"TensorFlow model loaded from {model_path}")
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
                
            logger.info(f"Search system fully loaded with {len(self.embeddings_db):,} embeddings and TensorFlow model!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading search system: {e}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert('RGB')
        image = image.resize((self.img_size, self.img_size))
        
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Get embedding for a single image
        
        Args:
            image: PIL Image
            
        Returns:
            np.ndarray: Image embedding
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_system() first.")
            
        img_array = self.preprocess_image(image)
        logger.info(f"Preprocessed image shape: {img_array.shape}, dtype: {img_array.dtype}, min: {img_array.min():.3f}, max: {img_array.max():.3f}")
        logger.info(f"EfficientNet preprocess_input applied: {img_array.mean():.3f} (should be ~0, not ~127)")
        
        embedding = self.model.predict(img_array, verbose=0)
        logger.info(f"Raw embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
        logger.info(f"Embedding stats - min: {embedding.min():.3f}, max: {embedding.max():.3f}, mean: {embedding.mean():.3f}")
        
        embedding_normalized = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        logger.info(f"L2 normalized embedding norm: {np.linalg.norm(embedding_normalized[0]):.3f}")
        
        return embedding_normalized[0]
    
    def search_similar_recipes(self, image: Image.Image, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.embeddings_db is None:
            raise ValueError("Embeddings database not loaded. Please call load_system() first.")
            
        query_embedding = self.get_image_embedding(image)
        query_embedding = query_embedding.reshape(1, -1)
        logger.info(f"Query embedding shape: {query_embedding.shape}, norm: {np.linalg.norm(query_embedding):.3f}")
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, self.embeddings_db)[0]
        logger.info(f"Similarities shape: {similarities.shape}, min: {similarities.min():.3f}, max: {similarities.max():.3f}, mean: {similarities.mean():.3f}")
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        logger.info(f"Top {top_k} similarities: {similarities[top_indices]}")
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            similarity = float(similarities[idx])
            image_path = self.image_paths[idx]
            
            result = {
                'rank': rank,
                'similarity': similarity,
                'image_path': image_path
            }
            
            if self.recipes_df is not None:
                try:
                    if idx < len(self.recipes_df):
                        recipe_data = self.recipes_df.iloc[idx % len(self.recipes_df)]
                        result.update({
                            'title': recipe_data.get('Title', 'Unknown Recipe'),
                            'ingredients': recipe_data.get('Ingredients', ''),
                            'instructions': recipe_data.get('Instructions', '')
                        })
                    else:
                        result.update({
                            'title': 'Recipe Details Unavailable',
                            'ingredients': '',
                            'instructions': ''
                        })
                except Exception as e:
                    logger.warning(f"Error accessing recipe data for idx {idx}: {e}")
                    result.update({
                        'title': 'Recipe Data Error',
                        'ingredients': '',
                        'instructions': ''
                    })
            else:
                result.update({
                    'title': 'Recipe Details Unavailable',
                    'ingredients': '',
                    'instructions': ''
                })
                
            results.append(result)
        
        return results 