# Brain Tumor Classifier - Instructions

Follow these steps to train the new model, start the web server, and test the application.

## 1. Train the New Model

1.  Open Terminal.
2.  Activate the environment:
    ```bash
    conda activate data_brain
    ```
3.  Navigate to the project directory:
    ```bash
    cd /path/to/your/project/data_brain_tumor
    ```
4.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
5.  In the browser, open `notebooks/train_v2_combined.ipynb`.
6.  Run all cells to train the model.
    *   This will train the model using both the original and external datasets with enhanced augmentation.
    *   The best model will be saved to `models/brain_tumor_resnet18_v2.pt`.

## 2. Start the Web Server

1.  Open a **new** Terminal window (or use the existing one if you stopped Jupyter).
    ```bash
    cd /path/to/your/project/data_brain_tumor
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    python3 -m website.app
    ```

## Training (Optional)

To retrain the model:

1.  **Navigate to project root**:
    ```bash
    cd /path/to/your/project/data_brain_tumor
    ```
4.  Start the Flask app:
    ```bash
    python -m website.app
    ```
    *   You should see output indicating the server is running on `http://127.0.0.1:3000`.

## 3. Test the Application

1.  Open your web browser and go to [http://localhost:3000](http://localhost:3000).
2.  **Random Test**:
    *   Click the **🎲 Random Test Image** button.
    *   The app will randomly select an image from the testing datasets (`Testing` and `external_dataset/testing`).
    *   It will display the prediction, confidence, and Grad-CAM heatmap.
3.  **Feedback**:
    *   If the prediction is correct, click **✅ Richtig**.
    *   If incorrect, click **❌ Falsch**, select the correct class from the dropdown, and click **Absenden**.
    *   This feedback is saved for future training.
