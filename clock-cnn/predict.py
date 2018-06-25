from keras.models import model_from_json
import sys
import argparse

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--image", help="path to image")
    a.add_argument("--scriptFolder", help="script folder")
    args = a.parse_args()

if args.image is None:
    a.print_help()
    sys.exit(1)    

if args.scriptFolder is None:
    scriptFolder = ''
else:
    scriptFolder = args.scriptFolder + "/"

if args.image is not None:

    # load json and create model
    json_file = open(scriptFolder + 'classifier.json', 'r')
    loaded_classifier_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_classifier_json)
    # load weights into new model
    classifier.load_weights(scriptFolder + "model.h5")

    #Making new predictions from our trained model :
    import numpy as np
    from keras.preprocessing import image
    from PIL import Image
    import sys

    test_image = image.load_img(args.image, target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    result = classifier.predict(test_image)

    # training_set.class_indices

    if result[0][0] == 1:
        prediction = '0' #nu e ceas
    else:
        prediction = '1' #e ceas

    print(prediction)
