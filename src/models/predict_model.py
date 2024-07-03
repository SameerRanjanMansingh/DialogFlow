import numpy as np
sequence_length = 5


def generate_next_sentence(model, tokenizer, text_sequence, max_length=5, temperature=0.80):
    # Tokenize the input text sequence
    encoded_sequence = tokenizer.encode(text_sequence, add_special_tokens=True)
    
    # Initialize the generated sequence
    generated_sequence = list(encoded_sequence)
    
    # Generate tokens until max_length or end token is reached
    for _ in range(max_length):
        # Ensure the sequence is of the required length
        input_sequence = generated_sequence[-sequence_length:] # -5:
        input_sequence = np.pad(input_sequence, (sequence_length - len(input_sequence), 0), 'constant')
        
        # Reshape the input sequence for model prediction
        input_sequence = np.array(input_sequence).reshape((1, sequence_length))
        
        # Predict the next token probabilities
        predicted_token_probs = model.predict(input_sequence)
        
        # Apply temperature to token probabilities
        predicted_token_probs = np.log(predicted_token_probs[0]) / temperature
        predicted_token_probs = np.exp(predicted_token_probs)
        predicted_token_probs /= np.sum(predicted_token_probs)
        
        # Sample the next token based on the predicted probabilities
        next_token_index = np.random.choice(len(tokenizer), p=predicted_token_probs.ravel())
        
        # Append the next token to the generated sequence
        generated_sequence.append(next_token_index)
        
        # Break if the end token is generated
        if next_token_index == tokenizer.eos_token_id:
            break
    
    # Decode the generated sequence
    generated_sentence = tokenizer.decode(generated_sequence)
    
    return generated_sentence


