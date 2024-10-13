### Machine Learning System Design Documentation

#### Version: 1.0
**Date:** October 3, 2024  
**Author:** Jeremie, Atou & Senanou  
**Version History:**  
- **v1.0:** Initial system design.

---

## Table of Contents

1. [Overview](#overview)
2. [Use Case](#use-case)
3. [Architecture](#architecture)
4. [Model Selection and Training](#model-selection-and-training)
5. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
6. [Model Deployment](#model-deployment)
7. [Performance Monitoring and Data Drift Detection](#performance-monitoring-and-data-drift-detection)
8. [Security and Privacy Considerations](#security-and-privacy-considerations)
9. [Version Control and CI/CD](#version-control-and-cicd)
10. [Conclusion](#conclusion)
11. [Future Iterations and Improvements](#future-iterations-and-improvements)

---

### 1. Overview <a name="overview"></a>

This document outlines the design and development of an image classification system. The system focuses on classifying images from user uploads into predefined categories. A pretrained ResNet-50 model is deployed using Azure with FastAPI, providing an interface for users to upload images and receive classification results. The system also includes monitoring capabilities for tracking model performance and data drift.

---

### 2. Use Case <a name="use-case"></a>

The system allows users to upload images for automatic classification into predefined categories. Specifically:
- Users interact through a web-based interface built with Next.js.
- A backend using FastAPI and a machine learning model processes the user-uploaded images and returns the predicted category.

#### Target Audience:
- Researchers, educators, and developers seeking quick and accurate image classification for a variety of use cases, such as plant species identification, object recognition, or medical image analysis.

---

### 3. Architecture <a name="architecture"></a>

**Components:**
1. **Frontend:**
   - **Framework:** Next.js
   - **Interface:** A file upload interface for image classification requests.
   - **Communication:** Uses axios to send user-uploaded images to the backend.

2. **Backend:**
   - **Framework:** FastAPI
   - **APIs:**
     - `/upload`: Handles image uploads and returns classification results.
     - **Model Integration:** Communicates with a pretrained ResNet-50 model hosted on Azure for image classification.

3. **Model Deployment:**
   - **Azure App Service:** Deployed FastAPI service with the ResNet-50 model using Azure for compute resources.
   - **Pretrained Model:** Uses ResNet-50 for classification tasks.

4. **Monitoring:**
   - **Data Drift Detection:** Monitoring model performance using EvidentlyAI and custom scripts to detect shifts in input image distributions.
   - **Performance Metrics:** Accuracy, precision, recall, and drift metrics tracked using Azure Monitoring.

#### System Diagram:

```plaintext
[Frontend] (Next.js UI for file uploads) 
   |
[Backend] (FastAPI for handling uploads)
   |
[ResNet-50] (Model for image classification) 
   |
[Monitoring] (Evidently AI + Azure Monitoring for drift detection)
```

---

### 4. Model Selection and Training <a name="model-selection-and-training"></a>

- **Pretrained Model:** ResNet-50
- **Task:** Image classification
- **API:** FastAPI service uses the ResNet-50 model for classifying uploaded images.
  
- **Training Data:** Pretrained ResNet-50 on the ImageNet dataset. For specific use cases, additional training data could be provided to fine-tune the model for specific classes.

---

### 5. Data Collection and Preprocessing <a name="data-collection-and-preprocessing"></a>

- **Image Formats:** Accepts JPEG, PNG, and other standard image formats.
- **Preprocessing:**
  - Images are resized and normalized according to the ResNet-50 input specifications.
  - Images are converted into appropriate tensors for processing by the model.
  - Augmentations like rotation and cropping can be applied during training for improved generalization.

---

### 6. Model Deployment <a name="model-deployment"></a>

- **Infrastructure:**
  - **Azure App Service:** Deployed as a scalable cloud service.
  - **Docker:** Backend services are containerized for easy deployment.
  
- **CI/CD Pipeline:** Using GitHub Actions for automated testing, building, and deployment of the FastAPI service and the ResNet-50 model.

---

### 7. Performance Monitoring and Data Drift Detection <a name="performance-monitoring-and-data-drift-detection"></a>

- **Tools:** 
  - **Evidently AI:** Integrated to track model performance metrics such as accuracy, precision, recall, and F1 score.
  - **Azure Monitoring:** Used for tracking service uptime, latency, and resource usage.

- **Data Drift Monitoring:** 
  - Monitor shifts in the image data distribution over time.
  - Alerts are set up for significant drift, prompting a model retraining or update cycle.

---

### 8. Security and Privacy Considerations <a name="security-and-privacy-considerations"></a>

- **Authentication:**
  - JWT tokens issued at the `/token` endpoint.
  - Tokens are required for accessing image classification endpoints.
  
- **Data Security:**
  - Uploaded images are processed securely.
  - Temporary storage for image processing ensures no sensitive data is exposed long-term.
  - Encryption of data in transit and at rest.

- **Privacy:** Compliance with data privacy laws such as GDPR, ensuring user data is processed with consent.

---

### 9. Version Control and CI/CD <a name="version-control-and-cicd"></a>

- **Git:** The source code for the FastAPI service and frontend is maintained in a Git repository.
- **Versioning:** 
  - Feature branches are used for iterative updates.
  - Versions are tagged for specific iterations (e.g., v1.0, v1.1).
  
- **CI/CD Pipeline:**  
  - **GitHub Actions:** Automated tests are run on pull requests. If tests pass, the containerized services are automatically built and deployed to Azure App Service.

---

### 10. Conclusion <a name="conclusion"></a>

The initial version of the image classification system provides a functional integration of a pretrained ResNet-50 model, with built-in mechanisms for deployment, monitoring, and security. Future iterations will focus on optimizing model performance, expanding support for additional image formats, and enhancing system scalability.

---

### 11. Future Iterations and Improvements <a name="future-iterations-and-improvements"></a>

1. **Model Fine-Tuning:** Depending on performance metrics, fine-tune the ResNet-50 model for specific image classification tasks (e.g., medical or agricultural images).
2. **User Feedback Loop:** Implement feedback mechanisms to allow users to provide feedback on classification results, aiding in model improvement.
3. **Scalability Enhancements:** Introduce load balancing mechanisms for handling high-volume queries.
4. **New Image Formats:** Extend support to additional image formats, including TIFF and BMP.

---

#### Version: 1.1 (Planned)
- **Changes:**
  - Add support for more image formats (e.g., TIFF, BMP).
  - Improved user feedback integration for model fine-tuning.
  - Detailed logging of model performance metrics.