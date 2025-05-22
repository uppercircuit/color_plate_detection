import os
import zipfile
import tempfile
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image, UnidentifiedImageError
import csv
import time
import matplotlib.pyplot as plt
import seaborn as sns


class ColorClassifier:
    def __init__(self, data_dir: str, zip_path: str, batch_size: int = 32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.data_dir = data_dir
        self.zip_path = zip_path
        self.batch_size = batch_size
        self.train_transform = self._build_train_transform()
        self.infer_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = self._load_dataset()
        self.resnet, self.scaler, self.pca, self.knn = None, None, None, None
        self.test_predictions = None
        self.test_labels = None

    def _build_train_transform(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_dataset(self):
        return datasets.ImageFolder(self.data_dir, transform=self.train_transform)

    def _create_sampler_loader(self):
        targets_tensor = torch.tensor(self.dataset.targets)
        class_counts = torch.bincount(targets_tensor)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[self.dataset.targets]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler, num_workers=4)

    def extract_features(self):
        print("Extracting features...")
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        resnet = resnet.to(self.device).eval()

        data_loader = self._create_sampler_loader()

        all_feats, all_lbls = [], []
        with torch.no_grad():
            for imgs, lbls in data_loader:
                imgs = imgs.to(self.device)
                feats = resnet(imgs)
                all_feats.append(feats.cpu())
                all_lbls.append(lbls)

        self.resnet = resnet
        features_np = torch.cat(all_feats, dim=0).numpy()
        labels_np = torch.cat(all_lbls, dim=0).numpy()

        print("Original features shape:", features_np.shape)
        return features_np, labels_np

    def fit_model(self, features_np, labels_np):
        print("Fitting model...")
        
        # Add feature scaling for better performance
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_np)
        
        # Use explained variance for PCA
        self.pca = PCA(n_components=0.95, random_state=42)
        features_pca = self.pca.fit_transform(features_scaled)
        print(f"PCA components: {features_pca.shape[1]}")

        # Better SMOTE parameters
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        k_neighbors = min(3, np.min(counts) - 1) if np.min(counts) > 1 else 1
        
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        features_bal, labels_bal = smote.fit_resample(features_pca, labels_np)

        X_train, X_test, y_train, y_test = train_test_split(
            features_bal, labels_bal, test_size=0.2, random_state=42, stratify=labels_bal
        )

        # Improved KNN parameters
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine', n_jobs=-1)
        self.knn.fit(X_train, y_train)

        y_pred = self.knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Store predictions and labels for confusion matrix
        self.test_predictions = y_pred
        self.test_labels = y_test
        
        print(f"Improved KNN Accuracy: {acc * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.dataset.classes))
        
        # Generate and display confusion matrix
        self.plot_confusion_matrix()

    def plot_confusion_matrix(self, save_path="confusion_matrix.png", figsize=(10, 8)):
        """
        Plot and save confusion matrix
        """
        if self.test_predictions is None or self.test_labels is None:
            print("No test predictions available. Run fit_model() first.")
            return
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.test_labels, self.test_predictions)
        
        # Create the plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.dataset.classes, 
                    yticklabels=self.dataset.classes,
                    cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Color Classification', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved as '{save_path}'")
        plt.show()
        
        # Print additional metrics
        self.print_confusion_matrix_metrics(cm)

    def print_confusion_matrix_metrics(self, cm):
        """
        Print detailed metrics from confusion matrix
        """
        print("\n" + "="*50)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*50)
        
        # Calculate per-class metrics
        num_classes = len(self.dataset.classes)
        
        for i, class_name in enumerate(self.dataset.classes):
            tp = cm[i, i]  # True positives
            fp = cm[:, i].sum() - tp  # False positives
            fn = cm[i, :].sum() - tp  # False negatives
            tn = cm.sum() - tp - fp - fn  # True negatives
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{class_name}:")
            print(f"  True Positives: {tp}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
        
        # Overall accuracy
        total_correct = np.trace(cm)
        total_samples = cm.sum()
        accuracy = total_correct / total_samples
        print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Total Samples: {total_samples}")
        print(f"Correctly Classified: {total_correct}")
        print(f"Misclassified: {total_samples - total_correct}")

    def analyze_misclassifications(self):
        """
        Analyze which classes are most commonly confused with each other
        """
        if self.test_predictions is None or self.test_labels is None:
            print("No test predictions available. Run fit_model() first.")
            return
        
        cm = confusion_matrix(self.test_labels, self.test_predictions)
        
        print("\n" + "="*50)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*50)
        
        # Find the most common misclassifications
        misclassifications = []
        for i in range(len(self.dataset.classes)):
            for j in range(len(self.dataset.classes)):
                if i != j and cm[i, j] > 0:
                    misclassifications.append({
                        'true_class': self.dataset.classes[i],
                        'predicted_class': self.dataset.classes[j],
                        'count': cm[i, j],
                        'percentage': (cm[i, j] / cm[i, :].sum()) * 100
                    })
        
        # Sort by count (descending)
        misclassifications.sort(key=lambda x: x['count'], reverse=True)
        
        print("\nTop Misclassifications:")
        for i, misc in enumerate(misclassifications[:10]):  # Show top 10
            print(f"{i+1:2d}. {misc['true_class']} → {misc['predicted_class']}: "
                  f"{misc['count']} samples ({misc['percentage']:.1f}%)")

    def predict_image(self, image_path: str, n_augment: int = 5, confidence_threshold: float = 0.6):
        try:
            img = Image.open(image_path).convert('RGB')
        except (UnidentifiedImageError, FileNotFoundError) as e:
            return "Unreadable", 0.0

        # Test-time augmentation with more variations
        augment_transforms = [
            self.infer_transform,
            transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ColorJitter(brightness=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ColorJitter(contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]

        all_probs = []
        for transform in augment_transforms[:n_augment]:
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feats = self.resnet(img_tensor)
                feats_np = feats.cpu().numpy()
                feats_scaled = self.scaler.transform(feats_np)
                feat_pca = self.pca.transform(feats_scaled)
                probs = self.knn.predict_proba(feat_pca)
                all_probs.append(probs[0])

        avg_probs = np.mean(all_probs, axis=0)
        pred_idx = np.argmax(avg_probs)
        conf = avg_probs[pred_idx]

        if conf < confidence_threshold:
            return "Unknown", conf
        return self.dataset.classes[pred_idx], conf

    def run_inference_on_zip(self):
        print("Running inference on ZIP of images...")
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
        image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(temp_dir)
            for file in files
            if os.path.splitext(file)[1].lower() in image_extensions
        ]

        print(f"Found {len(image_paths)} images.")
        results = []
        start_time = time.time()

        for path in image_paths:
            try:
                label, confidence = self.predict_image(path, confidence_threshold=0.5)
                results.append((os.path.basename(path), label, round(confidence, 4)))
                print(f"{os.path.basename(path)} → {label} ({confidence*100:.2f}%)")
            except Exception as e:
                print(f"Error processing {path}: {e}")

        end_time = time.time()
        print(f"\nInference completed in {end_time - start_time:.2f} seconds.")
        shutil.rmtree(temp_dir)
        return results

    def save_results_to_csv(self, results, output_csv="inference_results.csv"):
        with open(output_csv, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "Predicted Label", "Confidence"])
            writer.writerows(results)
        print(f"All predictions saved to {output_csv}")

    def save_confusion_matrix_csv(self, output_csv="confusion_matrix.csv"):
        """
        Save confusion matrix data to CSV file
        """
        if self.test_predictions is None or self.test_labels is None:
            print("No test predictions available. Run fit_model() first.")
            return
        
        cm = confusion_matrix(self.test_labels, self.test_predictions)
        
        with open(output_csv, "w", newline='') as f:
            writer = csv.writer(f)
            # Write header
            header = ["True\\Predicted"] + self.dataset.classes
            writer.writerow(header)
            # Write confusion matrix rows
            for i, class_name in enumerate(self.dataset.classes):
                row = [class_name] + cm[i].tolist()
                writer.writerow(row)
        
        print(f"Confusion matrix saved to {output_csv}")


if __name__ == "__main__":
    data_dir = "/home/beta/aditya_shrikul_workspace/color_plate/color_detection/number_plates_clean"
    zip_path = "/home/beta/aditya_shrikul_workspace/color_plate/color_detection/colour(LPN).zip"

    classifier = ColorClassifier(data_dir, zip_path)
    features, labels = classifier.extract_features()
    classifier.fit_model(features, labels)
    
    # Analyze misclassifications
    classifier.analyze_misclassifications()
    
    # Save confusion matrix as CSV
    classifier.save_confusion_matrix_csv()
    
    # Run inference
    results = classifier.run_inference_on_zip()
    classifier.save_results_to_csv(results)
