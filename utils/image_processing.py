def preprocess_image(self, image, target_size=(224, 224)):
    """
    Preprocess text image using a TensorFlow model for enhanced preprocessing.
    
    Parameters:
        image (PIL.Image): Input image.
        target_size (tuple): Target size for resizing.
    
    Returns:
        PIL.Image: The processed image.
    """
    from tensorflow.keras.preprocessing import image as tf_image
    
    # Convert to tensor and preprocess
    if isinstance(image, Image.Image):
        # Resize to the model's expected input size (224x448)
        # Notice the order of dimensions must match what the model expects
        img_resized = image.resize((224, 448), Image.BILINEAR)
        
        # Convert to numpy array and add batch dimension
        img_array = tf_image.img_to_array(img_resized)
        
        # The model expects (None, 224, 448, 3) but img_array might be (448, 224, 3)
        # Make sure the dimensions are correct
        if img_array.shape[0] == 448 and img_array.shape[1] == 224:
            # Transpose if needed - though this shouldn't happen with PIL image resizing
            img_array = img_array.transpose(1, 0, 2)
            
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Check the shape to make sure it matches what the model expects
        if img_array.shape[1:] != (224, 448, 3):
            raise ValueError(f"Expected shape (batch, 224, 448, 3), got {img_array.shape}")
        
        # Run inference using the model's signatures (more reliable than direct calling)
        try:
            # Try using the model's signatures
            infer = self.tf_model.signatures['serving_default']
            output = infer(tf.constant(img_array))
            
            # The output might be in a different format depending on the model
            # Look for the output tensor in the returned dictionary
            output_tensor_name = list(output.keys())[0]
            output_tensor = output[output_tensor_name]
            
        except (KeyError, AttributeError):
            # Fallback to direct model call if signatures don't work
            output = self.tf_model(img_array)
            if isinstance(output, dict):
                output_tensor = list(output.values())[0]
            else:
                output_tensor = output
        
        # Convert output tensor to numpy array
        output_image = output_tensor.numpy()
        
        # Extract the first channel (assuming this is the desired output)
        # Handle different shapes of output
        if len(output_image.shape) == 4:  # (batch, height, width, channels)
            output_channel = output_image[0, :, :, 0]
        elif len(output_image.shape) == 3:  # (height, width, channels)
            output_channel = output_image[:, :, 0]
        else:
            raise ValueError(f"Unexpected output shape: {output_image.shape}")
        
        # Convert to PIL Image (normalize to 0-255 range)
        output_channel = (output_channel * 255).astype(np.uint8)
        processed_image = Image.fromarray(output_channel)
        
        # Resize to the target size while maintaining aspect ratio
        processed_image.thumbnail(target_size, Image.BILINEAR)
        
        # Convert to RGB for compatibility with the rest of the pipeline
        processed_image = processed_image.convert("RGB")
        
        return processed_image
    else:
        raise TypeError("Input must be a PIL Image")