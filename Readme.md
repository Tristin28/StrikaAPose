Note that for this project, python 3.12 + is required for the reason that typing's package is used.

## Running the prototype

1. Install the dependencies: `pip install -r requirements.txt`
2. (Only needed if the images change) rebuild the dataset CSV: `python -m src.datapipeline.Build_CSV`
3. Start the web app from the project root: `python -m src.app`
4. Open `http://127.0.0.1:5000` in the browser, allow camera access, and strike a pose.

The skeleton is detected live in the browser with MediaPipe. Every 5 seconds the current
pose is sent to the Flask `/predict` endpoint, which normalises it, searches the dataset with
the chosen similarity metric (Euclidean / Cosine / Manhattan), and returns the closest
matching image to display.
