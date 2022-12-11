def shallow_model_post_processing(output_data):
    output_timestamps = [str(e) for e in output_data.index]
    output_error = [e for e in output_data[output_data.keys()[0]]]
    output_threshold = max(output_error)
    output_error = [-e + output_threshold for e in output_error]
    return output_error, output_timestamps, output_threshold
