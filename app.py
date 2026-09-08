from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ----------------------------------------------------
# ANALYZE ONION IMAGE
# ----------------------------------------------------

def analyze_onion(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Resize image for faster processing
    image = cv2.resize(image, (500, 500))

    # Convert image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ------------------------------------------------
    # STEP 1: DETECT ONION REGION
    # ------------------------------------------------

    # Saturated coloured regions are likely to contain onion
    lower_onion = np.array([0, 25, 20])
    upper_onion = np.array([179, 255, 255])

    onion_mask = cv2.inRange(hsv, lower_onion, upper_onion)

    # Remove very bright background pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, non_white_mask = cv2.threshold(
        gray,
        245,
        255,
        cv2.THRESH_BINARY_INV
    )

    onion_mask = cv2.bitwise_and(
        onion_mask,
        non_white_mask
    )

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)

    onion_mask = cv2.morphologyEx(
        onion_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    onion_pixels = cv2.countNonZero(onion_mask)

    if onion_pixels == 0:
        return {
            "quality_score": 0,
            "size": "Unknown",
            "onion_color": "Unknown",
            "defect_level": "Unknown",
            "defect_percentage": 0,
            "grade": "Unable to Analyze",
            "recommendation": "UPLOAD CLEAR IMAGE",
            "message": "Onion could not be detected clearly."
        }

    # ------------------------------------------------
    # STEP 2: NATURAL PURPLE / RED ONION DETECTION
    # ------------------------------------------------

    # Red onion hue ranges in OpenCV HSV:
    # Red/purple can occur near both ends of hue range

    lower_red_1 = np.array([0, 40, 40])
    upper_red_1 = np.array([15, 255, 255])

    lower_red_2 = np.array([150, 40, 40])
    upper_red_2 = np.array([179, 255, 255])

    purple_red_mask_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )

    purple_red_mask_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )

    purple_red_mask = cv2.bitwise_or(
        purple_red_mask_1,
        purple_red_mask_2
    )

    purple_red_mask = cv2.bitwise_and(
        purple_red_mask,
        onion_mask
    )

    purple_pixels = cv2.countNonZero(
        purple_red_mask
    )

    purple_percentage = (
        purple_pixels / onion_pixels
    ) * 100


    # ------------------------------------------------
    # STEP 3: DETECT BLACK / DARK BROWN DEFECTS
    # ------------------------------------------------

    # Black / rotten regions
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([179, 120, 65])

    black_mask = cv2.inRange(
        hsv,
        lower_black,
        upper_black
    )


    # Brown rotten areas
    lower_brown = np.array([5, 50, 20])
    upper_brown = np.array([25, 255, 180])

    brown_mask = cv2.inRange(
        hsv,
        lower_brown,
        upper_brown
    )


    # Combine defect masks
    defect_mask = cv2.bitwise_or(
        black_mask,
        brown_mask
    )


    # Only count defects inside onion
    defect_mask = cv2.bitwise_and(
        defect_mask,
        onion_mask
    )


    # ------------------------------------------------
    # STEP 4: REMOVE NATURAL RED/PURPLE SKIN
    # ------------------------------------------------

    # IMPORTANT:
    # Natural red/purple onion colour must NOT be
    # considered as a defect.

    defect_mask[purple_red_mask > 0] = 0


    # Remove very small noise
    defect_mask = cv2.morphologyEx(
        defect_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )


    # ------------------------------------------------
    # STEP 5: CALCULATE DEFECT PERCENTAGE
    # ------------------------------------------------

    defect_pixels = cv2.countNonZero(
        defect_mask
    )

    defect_percentage = (
        defect_pixels / onion_pixels
    ) * 100

    defect_percentage = round(
        defect_percentage,
        1
    )


    # ------------------------------------------------
    # STEP 6: DETECT ONION SIZE
    # ------------------------------------------------

    contours, _ = cv2.findContours(
        onion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            largest_contour
        )

        if area < 20000:
            size = "Small"

        elif area < 60000:
            size = "Medium"

        else:
            size = "Large"

    else:
        size = "Medium"


    # ------------------------------------------------
    # STEP 7: DETECT ONION COLOUR
    # ------------------------------------------------

    if purple_percentage > 10:
        onion_color = "Red / Purple Onion"
    else:
        onion_color = "Brown / Yellow Onion"


    # ------------------------------------------------
    # STEP 8: QUALITY SCORE
    # ------------------------------------------------

    # Defect percentage reduces score.
    # Healthy onion starts close to 100.

    quality_score = 100 - (
        defect_percentage * 2.5
    )

    # Prevent score going below 0
    quality_score = max(
        0,
        min(100, quality_score)
    )

    quality_score = round(
        quality_score,
        1
    )


    # ------------------------------------------------
    # STEP 9: DEFECT LEVEL
    # ------------------------------------------------

    if defect_percentage < 10:

        defect_level = "Low Defect"

    elif defect_percentage < 30:

        defect_level = "Moderate Defect"

    else:

        defect_level = "High Defect"


    # ------------------------------------------------
    # STEP 10: FINAL GRADE + RECOMMENDATION
    # ------------------------------------------------

    # GRADE A
    if (
        quality_score >= 80
        and defect_percentage < 10
    ):

        grade = "Grade A"
        recommendation = "BUY"
        message = (
            "Good Quality Onion - "
            "Suitable for Purchase"
        )


    # GRADE B
    elif (
        quality_score >= 50
        and defect_percentage < 30
    ):

        grade = "Grade B"
        recommendation = "BUY"
        message = (
            "Acceptable Quality Onion - "
            "Minor Defects Detected"
        )


    # GRADE C
    else:

        grade = "Grade C / Rejected"
        recommendation = "DO NOT BUY"
        message = (
            "Poor Quality Onion - "
            "High Defect Detected"
        )


    # ------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------

    result = {
        "quality_score": quality_score,
        "size": size,
        "onion_color": onion_color,
        "defect_level": defect_level,
        "defect_percentage": defect_percentage,
        "grade": grade,
        "recommendation": recommendation,
        "message": message
    }

    return result


# ----------------------------------------------------
# HOME PAGE
# ----------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    image_url = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "index.html",
                result=None,
                error="No image uploaded."
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                result=None,
                error="Please select an image."
            )

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        result = analyze_onion(
            filepath
        )

        image_url = (
            "uploads/" + filename
        )

    return render_template(
        "index.html",
        result=result,
        image_url=image_url
    )


# ----------------------------------------------------
# RUN APPLICATION
# ----------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )