"""
Function to determine required software on the PC.
"""
if __name__ == "__main__":
    try:
        import keras
        print("KERAS OK: "+ keras.__version__)
    except:
        print("Missing KERAS, required at least KERAS 2.0.8 plus Tensorflow as backend.")

    try:
        import tensorflow as tf
        print("Tensodflow OK: " + tf.__version__)
    except:
        print("Missing Tensorflow, required at least Tensorflow 1.2.1")
