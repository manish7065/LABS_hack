import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Load the group photo and the known image of Joe Biden
group_photo = face_recognition.load_image_file("group.jpg")
ayush_image = face_recognition.load_image_file("ayush.jpg")
krishna_image = face_recognition.load_image_file("krishna.jpg")


# Get the face encodings for the group photo and the known image of Joe Biden
group_photo_encoding = face_recognition.face_encodings(group_photo)[0]
ayush_encoding = face_recognition.face_encodings(ayush_image)[0]
krishna_encoding = face_recognition.face_encodings(krishna_image)[0]


# Find the location of all the faces in the group photo
face_locations = face_recognition.face_locations(group_photo)

# Compare the face encodings to see if any of the faces in the group photo match the known image of Joe Biden
matches = face_recognition.compare_faces([ayush_encoding], group_photo_encoding)

# Draw a circle around the matched face and display the name
if True in matches:
    match_index = matches.index(True)
    top, right, bottom, left = face_locations[match_index]
    image = Image.fromarray(group_photo)
    # image = image.resize((50, 30), resample=Image.BICUBIC)

    draw = ImageDraw.Draw(image)
    # draw.ellipse(((left, top), (right, bottom)), fill='red', outline ='red')
    draw.ellipse(((left, top), (right, bottom)), outline ='red', width = 5)
    # draw.text((left, bottom), "ayush", fill='red')
    
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf", 74)
    draw.text((left, bottom), "ayush", fill='red', font=font)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
