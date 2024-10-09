def get_new_model(model_orig, INPUT_DIM=32, INPUT_RESOLUTION=32**2, DIM_SIGNATURE=256, DIM_HASH=24, num_classes=None):
    # Create a deep copy of the original model to avoid modifying it directly
    model_new = deepcopy(model_orig)

    # Calculate the number of additional channels needed
    ## this is an integer, in my toy example I can fit all the hash and sig bits in one channel
    num_signature_channels = (DIM_SIGNATURE // INPUT_RESOLUTION) + 1
    num_hash_channels = (DIM_HASH // INPUT_RESOLUTION) + 1
    total_additional_channels = num_signature_channels + num_hash_channels

    # Update the first convolutional layer to accept additional channels
    in_channels_new = 3 + total_additional_channels
    model_new.conv1 = nn.Conv2d(
        in_channels_new, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    # Initialize the new conv1 weights
    with torch.no_grad():
        # Get pre-trained conv1 weights from the original model
        pre_trained_weights = model_orig.conv1.weight  # Shape: [64, 3, 3, 3]

        # Create new conv1 weights with additional channels
        new_weights = torch.zeros(64, in_channels_new, 3, 3)
        new_weights[:, :3, :, :] = pre_trained_weights  # Copy existing weights

        # Initialize the additional channels (e.g., with zeros or random weights)
        # For zeros:
        # new_weights[:, 3:, :, :] remains zeros

        # Alternatively, initialize with random weights:
        # nn.init.kaiming_normal_(new_weights[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        # Assign the new weights to conv1
        model_new.conv1.weight = nn.Parameter(new_weights)
        if num_classes is not None:
            # Modify the final linear layer to match the new number of classes
            in_features = model_new.linear.in_features  # Typically 512 for ResNet-18
            model_new.linear = nn.Linear(in_features, num_classes)

            # Initialize the new linear layer
            nn.init.kaiming_normal_(model_new.linear.weight, mode='fan_out', nonlinearity='relu')
            if model_new.linear.bias is not None:
                nn.init.constant_(model_new.linear.bias, 0)

    return model_new



def get_input(x, signature, hash_x, INPUT_RESOLUTION=32**2):
    # Get the concatenated input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if signature.shape[1] < INPUT_RESOLUTION:
        # Pad signature and hash with 0
        signature = torch.cat((signature, torch.zeros(
            signature.shape[0], INPUT_RESOLUTION-signature.shape[1]).to(signature.device)), dim=1)
        hash_x = torch.cat((hash_x, torch.zeros(
            hash_x.shape[0], INPUT_RESOLUTION-hash_x.shape[1]).to(hash_x.device)), dim=1)

    # Pad signature and hash to the next multiple of INPUT_RESOLUTION
    signature = torch.cat((signature, torch.zeros(
        signature.shape[0], INPUT_RESOLUTION-signature.shape[1]).to(signature.device)), dim=1)
    hash_x = torch.cat((hash_x, torch.zeros(
        hash_x.shape[0], INPUT_RESOLUTION-hash_x.shape[1]).to(signature.device)), dim=1)

    # Make signature and hash to be of shape (batch_size, (sqrt(INPUT_RESOLUTION)), sqrt(INPUT_RESOLUTION))
    signature = signature.view(
        signature.shape[0], -1, int(INPUT_RESOLUTION**0.5), int(INPUT_RESOLUTION**0.5))
    hash_x = hash_x.view(
        hash_x.shape[0], -1,  int(INPUT_RESOLUTION**0.5), int(INPUT_RESOLUTION**0.5))

    return torch.cat((x, signature, hash_x), dim=1).to(device)