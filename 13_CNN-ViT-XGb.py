import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import librosa
import warnings
warnings.filterwarnings('ignore')

# Custom attention layer for ViT
class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=12):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")
        
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Feature extraction functions
def extract_handcrafted_features(audio_data, sr=12000):
    """Extract 68 handcrafted features as mentioned in the paper"""
    features = []
    
    # Time-domain features
    features.append(np.mean(audio_data))  # Mean
    features.append(np.std(audio_data))   # Standard deviation
    features.append(np.max(audio_data))   # Maximum
    features.append(np.min(audio_data))   # Minimum
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    features.extend([np.mean(zcr), np.std(zcr)])
    
    # Energy and entropy features
    energy = np.sum(audio_data ** 2)
    features.append(energy)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    spectral_flux = np.diff(np.abs(librosa.stft(audio_data)), axis=1)
    
    features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
    features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
    features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
    features.extend([np.mean(spectral_flux), np.std(spectral_flux)])
    
    # MFCCs (13 coefficients + 13 delta)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    
    for i in range(13):
        features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
        features.extend([np.mean(delta_mfccs[i]), np.std(delta_mfccs[i])])
    
    # Chroma features (12 bins)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    for i in range(12):
        features.extend([np.mean(chroma[i]), np.std(chroma[i])])
    
    return np.array(features[:68])  # Ensure exactly 68 features

def create_spectro_temporal_tensor(audio_data, sr=12000):
    """Create multi-channel 2D tensor for ViT"""
    # MFCC spectrogram
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=128, n_fft=2048, hop_length=512)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=2048, hop_length=512)
    chroma = np.repeat(chroma, 128//12, axis=0)[:128]  # Repeat to match MFCC dimensions
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Combine into 3-channel tensor (like RGB image)
    spectro_tensor = np.stack([mfcc, chroma, mel_spec_db], axis=-1)
    
    # Resize to fixed dimensions (224x224 for ViT compatibility)
    from tf.image import resize # type: ignore
    spectro_tensor = resize(spectro_tensor[..., np.newaxis], [224, 224])
    spectro_tensor = tf.squeeze(spectro_tensor, axis=-1)
    
    return spectro_tensor.numpy()

# CNN Model for handcrafted features
def create_cnn_model(input_dim=68):
    """CNN for 68-dimensional handcrafted features"""
    inputs = Input(shape=(input_dim, 1))
    
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output embedding (not classification)
    outputs = Dense(128, name='cnn_embedding')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Vision Transformer for spectro-temporal data
def create_vit_model(input_shape=(224, 224, 3), patch_size=16, embed_dim=768, num_heads=12, num_layers=6):
    """Vision Transformer for spectro-temporal analysis"""
    inputs = Input(shape=input_shape)
    
    # Patch embedding
    patches = tf.image.extract_patches(
        images=inputs,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    patch_dim = patch_size * patch_size * input_shape[2]
    
    patches = tf.reshape(patches, [-1, num_patches, patch_dim])
    
    # Linear projection of patches
    encoded_patches = Dense(embed_dim)(patches)
    
    # Add position embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = Embedding(input_dim=num_patches, output_dim=embed_dim)(positions)
    encoded_patches = encoded_patches + position_embedding
    
    # Add CLS token
    cls_token = tf.Variable(tf.random.normal([1, 1, embed_dim]))
    cls_token = tf.tile(cls_token, [tf.shape(encoded_patches)[0], 1, 1])
    encoded_patches = tf.concat([cls_token, encoded_patches], axis=1)
    
    # Transformer blocks
    x = encoded_patches
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, embed_dim * 4)(x)
    
    # Extract CLS token
    representation = LayerNormalization(epsilon=1e-6)(x)
    cls_output = representation[:, 0]
    
    # Output embedding
    outputs = Dense(768, name='vit_embedding')(cls_output)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# XGBoost for metadata
def create_xgboost_model(metadata_features):
    """XGBoost model for structured metadata"""
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=1.0,
        objective='binary:logistic',
        eval_metric='auc',
        early_stopping_rounds=10,
        random_state=42
    )
    return xgb_model

# LLaMA-2 Reasoning Layer
class LlamaReasoningLayer:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        """Initialize LLaMA-2 for symbolic reasoning"""
        # Note: You'll need to configure access to LLaMA-2 model
        # This is a placeholder implementation
        self.tokenizer = None  # LlamaTokenizer.from_pretrained(model_name)
        self.model = None      # LlamaForCausalLM.from_pretrained(model_name)
        
    def format_prompt(self, cnn_embedding, vit_embedding, xgb_prediction, metadata_summary):
        """Format input for LLaMA-2 reasoning"""
        prompt = f"""
        Medical Audio Analysis Task:
        
        CNN Statistical Features (128-dim): {cnn_embedding.tolist()[:10]}... (truncated)
        Vision Transformer Spectro-temporal (768-dim): {vit_embedding.tolist()[:10]}... (truncated)
        XGBoost Metadata Prediction: {xgb_prediction}
        Patient Context: {metadata_summary}
        
        Based on the multimodal analysis above, provide:
        1. Binary classification: "COVID-19 Positive" or "Healthy"
        2. Confidence score (0-1)
        3. Brief clinical reasoning
        
        Response format:
        Classification: [COVID-19 Positive/Healthy]
        Confidence: [0-1]
        Reasoning: [Brief explanation linking acoustic and clinical features]
        """
        return prompt
    
    def reason(self, cnn_embedding, vit_embedding, xgb_prediction, metadata_summary):
        """Perform symbolic reasoning (simplified version)"""
    
        
        # Combine predictions with weighted average
        cnn_score = np.mean(cnn_embedding)
        vit_score = np.mean(vit_embedding) 
        xgb_score = xgb_prediction
        
        final_score = 0.3 * abs(cnn_score) + 0.4 * abs(vit_score) + 0.3 * xgb_score
        
        if final_score > 0.5:
            classification = "COVID-19 Positive"
            confidence = min(final_score, 0.95)
        else:
            classification = "Healthy"
            confidence = min(1 - final_score, 0.95)
            
        reasoning = f"Multimodal analysis shows acoustic patterns consistent with {classification.lower()} classification based on spectro-temporal dynamics and clinical metadata."
        
        return {
            'classification': classification,
            'confidence': confidence,
            'reasoning': reasoning,
            'binary_prediction': 1 if classification == "COVID-19 Positive" else 0
        }

# Main multimodal framework
class MultimodalCOVIDDetection:
    def __init__(self):
        self.cnn_model = None
        self.vit_model = None
        self.xgb_model = None
        self.llama_reasoner = LlamaReasoningLayer()
        self.scaler = StandardScaler()
        
    def prepare_data(self, audio_data, metadata, sr=12000):
        """Prepare data for all three branches"""
        # CNN branch: handcrafted features
        cnn_features = []
        for audio in audio_data:
            features = extract_handcrafted_features(audio, sr)
            cnn_features.append(features)
        cnn_features = np.array(cnn_features).reshape(-1, 68, 1)
        
        # ViT branch: spectro-temporal tensors
        vit_features = []
        for audio in audio_data:
            tensor = create_spectro_temporal_tensor(audio, sr)
            vit_features.append(tensor)
        vit_features = np.array(vit_features)
        
        # XGBoost branch: metadata
        metadata_scaled = self.scaler.fit_transform(metadata)
        
        return cnn_features, vit_features, metadata_scaled
    
    def train(self, audio_data, metadata, labels, num_folds=5):
        """Train all components with cross-validation"""
        
        # Prepare data
        cnn_features, vit_features, metadata_scaled = self.prepare_data(audio_data, metadata)
        
        # Cross-validation setup
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=75)
        
        fold_results = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'auc': [], 'specificity': []
        }
        
        fold_no = 1
        
        for train_idx, val_idx in kfold.split(cnn_features, labels):
            print(f'Training fold {fold_no}/{num_folds}...')
            
            # Split data
            cnn_train, cnn_val = cnn_features[train_idx], cnn_features[val_idx]
            vit_train, vit_val = vit_features[train_idx], vit_features[val_idx]
            meta_train, meta_val = metadata_scaled[train_idx], metadata_scaled[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Train CNN model
            self.cnn_model = create_cnn_model()
            cnn_classifier = Model(inputs=self.cnn_model.input, 
                                 outputs=Dense(1, activation='sigmoid')(self.cnn_model.output))
            cnn_classifier.compile(optimizer='adamax', loss='binary_crossentropy', 
                                 metrics=['accuracy', 'precision', 'recall'])
            
            cnn_classifier.fit(cnn_train, y_train, epochs=50, batch_size=16, 
                             validation_data=(cnn_val, y_val), verbose=0)
            
            # Train ViT model  
            self.vit_model = create_vit_model()
            vit_classifier = Model(inputs=self.vit_model.input,
                                 outputs=Dense(1, activation='sigmoid')(self.vit_model.output))
            vit_classifier.compile(optimizer='adamax', loss='binary_crossentropy',
                                 metrics=['accuracy', 'precision', 'recall'])
            
            vit_classifier.fit(vit_train, y_train, epochs=30, batch_size=8,
                             validation_data=(vit_val, y_val), verbose=0)
            
            # Train XGBoost model
            self.xgb_model = create_xgboost_model(meta_train)
            self.xgb_model.fit(meta_train, y_train, 
                             eval_set=[(meta_val, y_val)], verbose=False)
            
            # Get embeddings and predictions
            cnn_embeddings = self.cnn_model.predict(cnn_val, verbose=0)
            vit_embeddings = self.vit_model.predict(vit_val, verbose=0)
            xgb_predictions = self.xgb_model.predict_proba(meta_val)[:, 1]
            
            # LLaMA-2 reasoning
            final_predictions = []
            for i in range(len(cnn_embeddings)):
                metadata_summary = f"Age: {metadata_scaled[val_idx[i]][0]}, Gender: {metadata_scaled[val_idx[i]][1]}"
                
                reasoning_result = self.llama_reasoner.reason(
                    cnn_embeddings[i], vit_embeddings[i], 
                    xgb_predictions[i], metadata_summary
                )
                final_predictions.append(reasoning_result['binary_prediction'])
            
            final_predictions = np.array(final_predictions)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, final_predictions)
            precision = precision_score(y_val, final_predictions, zero_division=0)
            recall = recall_score(y_val, final_predictions)
            f1 = f1_score(y_val, final_predictions)
            auc = roc_auc_score(y_val, final_predictions)
            
            # Calculate specificity
            tn = np.sum((y_val == 0) & (final_predictions == 0))
            fp = np.sum((y_val == 0) & (final_predictions == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            fold_results['accuracy'].append(accuracy)
            fold_results['precision'].append(precision)
            fold_results['recall'].append(recall)
            fold_results['f1'].append(f1)
            fold_results['auc'].append(auc)
            fold_results['specificity'].append(specificity)
            
            print(f"Fold {fold_no} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print("-" * 50)
            
            fold_no += 1
        
        # Print overall results
        print("\nOverall Results:")
        for metric, values in fold_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"Mean {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
        
        return fold_results

# Example usage
if __name__ == "__main__":

    covid_detector = MultimodalCOVIDDetection()
    # Load your COUGHVID dataset here
    # audio_data, metadata, labels = load_your_data_function()    
    print("Multimodal COVID-19 Detection Framework Ready!")
    print("Replace the example data loading with your actual COUGHVID dataset.")