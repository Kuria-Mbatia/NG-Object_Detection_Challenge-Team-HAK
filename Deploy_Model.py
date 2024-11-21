import os
import sys
import numpy as np
import cv2
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from scipy.ndimage import distance_transform_edt



CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
               'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 
               'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



class OptimizedCocoConfig(Config):
    """Optimized configuration for better Mask R-CNN accuracy"""
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81
    
    
    IMAGE_MIN_DIM = 832  
    IMAGE_MAX_DIM = 1344 
    
    # Enhanced backbone
    BACKBONE = "resnet101"
    
    # Optimized RPN settings for dense scenarios
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 1
    RPN_NMS_THRESHOLD = 0.8
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512
    
    # Optimized ROI settings for overlapping objects
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    DETECTION_MAX_INSTANCES = 300
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.3
    
    # Image padding mode
    IMAGE_PADDING = True  #maintain aspect retio
    
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def soft_nms(boxes, scores, method='gaussian', sigma=0.5, thresh=0.3):
    """Soft-NMS implementation for dense object detection"""
    N = len(scores)
    indices = list(range(N))
    
    for i in range(N):
        maxscore = scores[i]
        maxpos = i
        
        tx1, ty1, tx2, ty2 = boxes[i]
        ts = scores[i]
        
        pos = i + 1
        while pos < N:
            if scores[pos] > maxscore:
                maxscore = scores[pos]
                maxpos = pos
            pos += 1
        
        # Swap
        boxes[i], boxes[maxpos] = boxes[maxpos].copy(), boxes[i].copy()
        scores[i], scores[maxpos] = scores[maxpos], scores[i]
        indices[i], indices[maxpos] = indices[maxpos], indices[i]
        
        for pos in range(i + 1, N):
            iou = compute_iou(boxes[i], boxes[pos])
            
            if method == 'gaussian':
                scores[pos] *= np.exp(-(iou * iou) / sigma)
            else:  # linear
                if iou > thresh:
                    scores[pos] *= (1 - iou)
    
    # Filter out boxes with low scores
    keep = scores > thresh
    return boxes[keep], scores[keep], np.array(indices)[keep]

def apply_test_time_augmentation(model, image):
    """Enhanced test-time augmentation with better handling of dense objects"""
    augmented_predictions = []
    
    #original prediction
    original_pred = model.detect([image], verbose=0)[0]
    augmented_predictions.append(original_pred)
    
    # Horizontal flip with proper coordinate adjustment
    flipped_h = np.fliplr(image)
    pred_h = model.detect([flipped_h], verbose=0)[0]
    w = image.shape[1]
    pred_h['rois'][:, [0, 2]] = w - pred_h['rois'][:, [2, 0]]
    pred_h['masks'] = np.fliplr(pred_h['masks'])
    augmented_predictions.append(pred_h)
    
    # Multi-scale testing
    scales = [0.8, 1.2]  # Test at different scales
    for scale in scales:
        scaled_dim = tuple(int(x * scale) for x in image.shape[:2])
        scaled_image = cv2.resize(image, (scaled_dim[1], scaled_dim[0]))
        pred_s = model.detect([scaled_image], verbose=0)[0]
        
        # Rescale predictions to original size
        scale_factor = np.array([image.shape[0]/scaled_dim[0], 
                               image.shape[1]/scaled_dim[1]])
        pred_s['rois'] = pred_s['rois'] * np.tile(scale_factor, 2)
        pred_s['masks'] = cv2.resize(pred_s['masks'].astype(float), 
                                   (image.shape[1], image.shape[0]))
        augmented_predictions.append(pred_s)
    
    # Combine predictions using enhanced NMS
    all_masks = np.concatenate([p['masks'] for p in augmented_predictions], axis=-1)
    all_scores = np.concatenate([p['scores'] for p in augmented_predictions])
    all_class_ids = np.concatenate([p['class_ids'] for p in augmented_predictions])
    all_rois = np.concatenate([p['rois'] for p in augmented_predictions], axis=0)
    
    final_predictions = {
        'masks': [],
        'rois': [],
        'class_ids': [],
        'scores': []
    }
    
    # Process each class separately
    for class_id in np.unique(all_class_ids):
        class_mask = all_class_ids == class_id
        if not np.any(class_mask):
            continue
        
        class_rois = all_rois[class_mask]
        class_scores = all_scores[class_mask]
        class_masks = all_masks[:, :, class_mask]
        
        # Apply Soft-NMS
        refined_rois, refined_scores, keep = soft_nms(
            class_rois, 
            class_scores,
            method='gaussian',
            sigma=0.5,
            thresh=0.3
        )
        
        # Handle overlapping instances
        refined_masks = handle_overlapping_instances(
            class_masks[:, :, keep],
            refined_scores
        )
        
        #final predictions
        final_predictions['masks'].append(refined_masks)
        final_predictions['rois'].append(refined_rois)
        final_predictions['scores'].extend(refined_scores)
        final_predictions['class_ids'].extend([class_id] * len(refined_scores))
    
    # Combine classes
    if final_predictions['masks']:
        final_predictions['masks'] = np.concatenate(final_predictions['masks'], axis=-1)
        final_predictions['rois'] = np.concatenate(final_predictions['rois'], axis=0)
        final_predictions['class_ids'] = np.array(final_predictions['class_ids'])
        final_predictions['scores'] = np.array(final_predictions['scores'])
    else:
        final_predictions['masks'] = np.zeros((image.shape[0], image.shape[1], 0))
        final_predictions['rois'] = np.zeros((0, 4))
        final_predictions['class_ids'] = np.zeros((0,))
        final_predictions['scores'] = np.zeros((0,))
    
    return final_predictions

def handle_overlapping_instances(masks, scores):
    """Handle overlapping instances using distance transform and watershed"""
    n_masks = masks.shape[-1]
    refined_masks = np.zeros_like(masks)
    
    for i in range(n_masks):
        mask = masks[:, :, i]
        # Apply distance transform
        dist_transform = distance_transform_edt(mask)
        
        # Find peaks in distance transform
        peak_thresh = 0.7 * dist_transform.max()
        peaks = dist_transform > peak_thresh
        
        # Use peaks as markers for watershed
        markers = np.zeros_like(mask)
        markers[peaks] = 1
        
        # Apply watershed using distance transform
        from skimage.segmentation import watershed
        labels = watershed(-dist_transform, markers, mask=mask)
        
        # Weight by detection score
        refined_masks[:, :, i] = (labels > 0) * scores[i]
    
    return refined_masks


def process_images(input_dir, output_dir, weights_path):
    """Process images with enhanced dense object detection"""
    config = OptimizedCocoConfig()
    
    #verify dim
    if config.IMAGE_MIN_DIM % 64 != 0 or config.IMAGE_MAX_DIM % 64 != 0:
        raise ValueError("Image dimensions must be divisible by 64")
    
    model = modellib.MaskRCNN(mode="inference", 
                             config=config,
                             model_dir=os.getcwd())
    
    print("Loading weights...")
    model.load_weights(weights_path, by_name=True)
    
    image_files = list(Path(input_dir).glob("*.jpg")) + \
                 list(Path(input_dir).glob("*.jpeg")) + \
                 list(Path(input_dir).glob("*.png"))
    
    print(f"Found {len(image_files)} images to process")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\nProcessing image {i}/{len(image_files)}: {image_path.name}")
        
        try:
            #read process..
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            #dim valid
            h, w = image.shape[:2]
            #dim calc
            new_h = ((h + 63) // 64) * 64
            new_w = ((w + 63) // 64) * 64
            

            if h != new_h or w != new_w:
                padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                padded_image[:h, :w] = image
                image = padded_image
            
            results = apply_test_time_augmentation(model, image)
            
            #pad remove
            if h != new_h or w != new_w:
                results['masks'] = results['masks'][:h, :w]
                #ROIS adjs
                scale_h, scale_w = h/new_h, w/new_w
                results['rois'] = results['rois'] * [scale_h, scale_w, scale_h, scale_w]
            
            save_visualization(image[:h, :w], results, image_path, output_dir)
            print(f"Successfully processed {image_path.name}")
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
        
        tf.keras.backend.clear_session()
    
    print("\nProcessing complete!")

def setup_output_directories(base_output_dir, image_name):
    """Create organized directory structure for outputs"""
    #base dir
    base_dir = Path(base_output_dir)
    base_dir.mkdir(exist_ok=True)
    
    #img dir
    image_dir = base_dir / Path(image_name).stem
    image_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    masks_dir = image_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    
    return image_dir, masks_dir
def apply_masks_to_image(image, masks, class_ids, class_names, scores, alpha=0.5):
    """
    Apply segmentation masks to the image with different colors for each class
    
    Args:
        image: RGB image array
        masks: [height, width, num_instances] Instance masks
        class_ids: [num_instances] Class IDs for each instance
        class_names: List of class names
        scores: [num_instances] Confidence scores
        alpha: Mask transparency (0=transparent, 1=opaque)
    """
    # Make a copy of the image to avoid modifying the original
    masked_image = image.copy()
    
    # Generate random colors for each class (consistent across instances)
    np.random.seed(1)
    class_colors = {class_id: np.random.randint(0, 255, 3) 
                   for class_id in np.unique(class_ids)}
    
    # Apply each instance mask
    for i in range(masks.shape[-1]):
        mask = masks[:, :, i]
        class_id = class_ids[i]
        score = scores[i]
        color = class_colors[class_id]
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        for c in range(3):
            colored_mask[:, :, c] = np.where(mask == 1, color[c], 0)
        
        # Blend mask with image
        mask_area = mask > 0
        masked_image[mask_area] = (alpha * colored_mask[mask_area] + 
                                 (1 - alpha) * image[mask_area]).astype(np.uint8)
        
        # Add text label
        label = f"{class_names[class_id]} ({score:.2f})"
        # Find top-left corner of mask
        y, x = np.nonzero(mask)
        if len(y) > 0 and len(x) > 0:
            text_pos = (int(np.min(x)), int(np.min(y)) - 5)
            cv2.putText(masked_image, label, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
    
    return masked_image

def save_visualization(image, results, image_path, output_dir):
    """Save detection results with masks applied to the image"""
    # Setup directories
    image_name = Path(image_path).name
    image_dir, masks_dir = setup_output_directories(output_dir, image_name)
    
    # Copy original image
    shutil.copy2(image_path, image_dir / "original_image.jpg")
    
    r = results
    
    # Apply masks to image
    masked_image = apply_masks_to_image(
        image, 
        r['masks'],
        r['class_ids'],
        CLASS_NAMES,
        r['scores']
    )
    
    # Save the masked image
    masked_image_path = image_dir / "masked_image.png"
    plt.imsave(str(masked_image_path), masked_image)
    
    # Create visualization with both masks and bounding boxes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image with bounding boxes and masks
    visualize.display_instances(
        image,
        r['rois'],
        r['masks'],
        r['class_ids'],
        CLASS_NAMES,
        r['scores'],
        ax=ax1,
        title="Detections with Bounding Boxes"
    )
    
    # Image with only masks applied
    ax2.imshow(masked_image)
    ax2.set_title("Segmentation Masks")
    ax2.axis('off')
    
    # Save the comparison visualization
    comparison_path = image_dir / "detection_comparison.png"
    plt.savefig(comparison_path, bbox_inches='tight')
    plt.close()
    
    # Save individual masks with colored overlays
    for i in range(len(r['class_ids'])):
        mask = r['masks'][:, :, i]
        class_name = CLASS_NAMES[r['class_ids'][i]]
        score = r['scores'][i]
        
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        color = np.random.randint(0, 255, 3)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        
        # Save both binary and colored masks
        mask_filename = f"mask_{i+1}_{class_name}_{score:.2f}"
        cv2.imwrite(str(masks_dir / f"{mask_filename}_binary.png"), 
                   (mask * 255).astype(np.uint8))
        cv2.imwrite(str(masks_dir / f"{mask_filename}_colored.png"), 
                   colored_mask)
    
    # Save detection summary
    summary_path = image_dir / "detection_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Detection Results for {image_name}\n")
        f.write("-" * 50 + "\n")
        for i in range(len(r['scores'])):
            class_name = CLASS_NAMES[r['class_ids'][i]]
            score = r['scores'][i]
            f.write(f"Object {i+1}: {class_name} (Score = {score:.2f})\n")

def main():
    # Configuration
    INPUT_DIR = "Test_Images"    
    OUTPUT_DIR = "Test_Images_Results"      
    WEIGHTS_PATH = "mask_rcnn_coco.h5"
    
    print(f"Processing images from: {INPUT_DIR}")
    print(f"Saving results to: {OUTPUT_DIR}")
    
    process_images(INPUT_DIR, OUTPUT_DIR, WEIGHTS_PATH)

if __name__ == "__main__":
    main()