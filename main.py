from fastapi import FastAPI, UploadFile, File, HTTPException
import traceback

app = FastAPI()


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File content is empty.")

        # Save to temporary path if necessary for cv2 reading
        with open("temp_image.jpg", "wb") as temp_file:
            temp_file.write(content)

        # Run the prediction (ensure your model path is correct)
        results = model.predict(source="temp_image.jpg", conf=0.01, show=True)

        # Format and return the response
        predictions = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id]

                predictions.append({
                    "label": label,
                    "confidence": float(confidence),
                    "box": [x1, y1, x2, y2]
                })

        return {"predictions": predictions}

    except Exception as e:
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        print(f"Error: {e}\nTraceback: {traceback_str}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
