# Importing necessary libraries
import streamlit as st  # Streamlit for building web app UI
import pandas as pd  # Pandas for data manipulation
import numpy as np  # Numpy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting
import seaborn as sns  # Seaborn for advanced visualizations

# Scikit-learn imports for machine learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # Data splitting and hyperparameter tuning
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Encoding categorical variables and scaling numeric features
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix  # Model evaluation metrics
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier

# App title and description
st.title("Stroke Prediction with Random Forest & Detailed Visualization")  # Main title of the app
st.markdown("Predict stroke risk and visualize results with detailed labeled charts and color coding.")  # Short description

# -------------------------------
# 1. Upload Dataset
# -------------------------------
st.sidebar.header("Upload Dataset")  # Sidebar header for file upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])  # File uploader widget accepting only CSV files

# Check if user uploaded a file
if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Read the CSV into a Pandas DataFrame
    st.subheader("Original Dataset Preview")  # Subheader for dataset preview
    st.dataframe(df.head())  # Display first 5 rows of dataset
    st.write("Dataset Shape:", df.shape)  # Display the shape (rows, columns)

    # -------------------------------
    # 2. Data Cleaning
    # -------------------------------
    df = df.drop_duplicates()  # Remove duplicate rows
    if "bmi" in df.columns:  # If BMI column exists
        df.drop(columns=["bmi"], inplace=True)  # Drop BMI column
    df = df.dropna(subset=['gender', 'stroke'])  # Drop rows where gender or stroke is missing

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns  # Select numeric columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)  # Replace NaN with median value

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns  # Select object (string) columns
    encoder = LabelEncoder()  # Initialize LabelEncoder
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])  # Convert categories to integers

    # -------------------------------
    # 3. Feature & Target Split
    # -------------------------------
    X = df.drop("stroke", axis=1)  # Features: all columns except target
    y = df["stroke"]  # Target variable: stroke

    # Split dataset into training and testing sets (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale features to standardize values (mean=0, std=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit scaler on training data and transform
    X_test = scaler.transform(X_test)  # Transform test data using same scaler

    # -------------------------------
    # 4. Model Training
    # -------------------------------
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)  # Initialize Random Forest with balanced class weights

    # Hyperparameter grid for RandomizedSearchCV
    param_dist = {
        "n_estimators": [100, 200, 300],  # Number of trees in the forest
        "max_depth": [5, 10, 15, None],  # Maximum depth of each tree
        "min_samples_split": [2, 5, 10],  # Minimum samples required to split a node
        "min_samples_leaf": [1, 2, 4]  # Minimum samples required at a leaf node
    }

    # Randomized search for hyperparameter tuning
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist,  # Model and parameters
        n_iter=10,  # Number of random combinations to try
        scoring="roc_auc",  # Optimize based on ROC AUC
        cv=3,  # 3-fold cross-validation
        random_state=42,  # Reproducibility
        n_jobs=-1,  # Use all CPU cores
        verbose=0  # No verbose output
    )

    # Train model with spinner for UI feedback
    with st.spinner("Training Random Forest..."):
        search.fit(X_train, y_train)  # Fit model on training data

    best_model = search.best_estimator_  # Get the best trained model
    st.success("Training Completed!")  # Show success message
    st.write("Best Parameters:", search.best_params_)  # Display best hyperparameters

    # -------------------------------
    # 5. Evaluation
    # -------------------------------
    y_pred = best_model.predict(X_test)  # Predicted labels for test data
    y_proba = best_model.predict_proba(X_test)[:,1]  # Predicted probabilities for positive class (stroke)

    st.subheader("Classification Report")  # Classification report title
    st.text(classification_report(y_test, y_pred))  # Display precision, recall, f1-score

    auc = roc_auc_score(y_test, y_proba)  # Calculate ROC AUC score
    st.write("AUC Score:", auc)  # Display AUC score

    # -------------------------------
    # 6. Detailed Charts with Labels, Legends & Colors
    # -------------------------------

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)  # Heatmap visualization
    ax.set_xlabel("Predicted Stroke Status")
    ax.set_ylabel("Actual Stroke Status")
    ax.set_title("Confusion Matrix: Actual vs Predicted\nBlue Shades indicate count of patients")
    ax.text(0.5, -0.3, "X-axis: Predicted (0=No Stroke, 1=Stroke)\nY-axis: Actual (0=No Stroke, 1=Stroke)",
            fontsize=10, ha="center", transform=ax.transAxes)
    st.pyplot(fig)  # Display plot in Streamlit
    st.markdown("- **Top-left**: Correctly predicted no stroke\n- **Bottom-right**: Correctly predicted stroke\n- **Off-diagonal**: Incorrect predictions")

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)  # Compute FPR and TPR
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color="darkorange", linewidth=2)  # ROC curve
    ax.plot([0,1], [0,1], linestyle="--", color="grey", label="Random Guess")  # Baseline
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve: True Positive Rate vs False Positive Rate")
    ax.legend()
    st.pyplot(fig)
    st.markdown("**ROC curve interpretation**: The closer the orange curve is to the top-left corner, the better the model.")

    # Feature Importance
    importances = best_model.feature_importances_  # Get importance score of each feature
    features = X.columns  # Feature names
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=importances, y=features, palette="coolwarm", ax=ax)  # Horizontal bar plot
    ax.set_xlabel("Importance Score (Higher = More Impact on Stroke Prediction)")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance by Random Forest")
    st.pyplot(fig)
    st.markdown("**Interpretation**: Longer bars indicate features that contribute more to prediction.")

    # Stroke Count by Gender
    st.subheader("Stroke Count by Gender")
    unique_stroke = df['stroke'].unique()  # Unique stroke values (0,1)
    palette_dict = {val: "salmon" if val==1 else "skyblue" for val in unique_stroke}  # Color mapping
    fig, ax = plt.subplots()
    sns.countplot(x="gender", hue="stroke", data=df, palette=palette_dict, ax=ax)  # Countplot by gender
    ax.set_xlabel("Gender of Patient")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Number of Stroke vs No Stroke Patients by Gender")
    ax.legend(title="Stroke Status", labels=["No Stroke (0)","Stroke (1)"])
    st.pyplot(fig)
    st.markdown("- Skyblue = No Stroke\n- Salmon = Stroke")

    # Age Distribution by Stroke
    if "age" in df.columns:
        st.subheader("Age Distribution by Stroke")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="age", hue="stroke", multiple="stack", bins=30,
                     palette=palette_dict, ax=ax)  # Histogram by age
        ax.set_xlabel("Age of Patient (Years)")
        ax.set_ylabel("Number of Patients")
        ax.set_title("Distribution of Ages for Stroke vs No Stroke")
        ax.legend(title="Stroke Status", labels=["No Stroke (0)","Stroke (1)"])
        st.pyplot(fig)
        st.markdown("- Shows which age groups have higher stroke prevalence.")

    # Predicted Stroke Probability Distribution
    st.subheader("Predicted Stroke Probability Distribution")
    X_scaled = scaler.transform(X)  # Scale full dataset
    y_proba_all = best_model.predict_proba(X_scaled)[:,1]  # Predict probabilities
    fig, ax = plt.subplots()
    sns.histplot(y_proba_all, bins=20, kde=True, color="mediumorchid", ax=ax)  # Probability distribution plot
    ax.set_xlabel("Predicted Stroke Probability (0=Low, 1=High)")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Distribution of Predicted Stroke Probabilities")
    st.pyplot(fig)
    st.markdown("- **Higher probability = higher predicted risk of stroke**\n- KDE curve shows overall risk distribution.")

    # -------------------------------
    # 7. Predictions for Whole Dataset
    # -------------------------------
    y_pred_all = best_model.predict(X_scaled)  # Predict classes for full dataset
    results = df.copy()  # Copy original dataset
    results["Predicted_Stroke"] = y_pred_all  # Add predicted stroke column
    results["Stroke_Probability"] = y_proba_all  # Add predicted probability column

    st.subheader("Full Dataset with Predictions")
    st.dataframe(results)  # Display results

    # Allow user to download predictions as CSV
    csv = results.to_csv(index=False).encode('utf-8')  # Convert to CSV
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="stroke_predictions_full.csv",
        mime="text/csv"
    )
