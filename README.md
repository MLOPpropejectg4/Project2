Here is an updated README focused on image classification, removing the chat part:

---

# Okapi Classifier: An Image Classification Web Application
## Product Design

This project provides a platform where users can upload images and get them classified into predefined categories using machine learning models. The platform utilizes a **pretrained model** for the classification task. The project is built using Python, FastAPI, and Docker.

## Features
- **Image Classification**: Upload images in supported formats (JPG, PNG) and receive predictions based on a pretrained deep learning model.
- **Multiple Categories**: Support for multiple classification categories based on the dataset used to train the model.
- **API Access**: REST API for easy integration with other services or custom applications.

## Requirements
```txt
python>=3.8
torch
torchvision
fastapi
uvicorn
pytest
python-dotenv
bcrypt 
```

Install dependencies locally with:
```bash
pip install -r requirements.txt
```

## Project Structure

```plaintext
.
├── app                                          # FastAPI app
│   ├── __init__.py
│   ├── app.py
│   └── main.py
├── mockup                                       # Frontend mockup
│   ├── SignUp.png
│   ├── accountSuccessfullyCreated.png
│   ├── login.png
│   └── updateUserInfoConfirmation.png
├── tests                                        # Test functions
│   ├── __init__.py
│   ├── sample.jpg                               # Sample image for testing
│   ├── test_auth.py
│   ├── test_classification.py                   # Test functions for image classification
└── utils                                        # Utility functions
|   ├── __init__.py
|   ├── auth.py                                  # Handles user authentication
|   ├── image_classification.py                  # Image classification function
├── Dockerfile                                   # Docker configuration
├── requirements.txt                             # Dependencies
├── README.md

```

## Installation
To run the application, you need to install the required dependencies. You can install them locally or use Docker. Here are the steps for local installation:

1. Clone the repository:
```bash
git clone https://github.com/jnlandu/image-classification.git
```
2. Navigate to the project directory:
```bash
cd image-classification
```
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
4. Run the FastAPI application:
```bash
uvicorn app.main:app --reload
```
Note: You will need to set environment variables in the `.env` file for any configurations, such as database credentials.

## Usage
Once the application is running, you can access the FastAPI Swagger UI at `http://localhost:8000/docs`. Here are the steps to use the app:

1. Authenticate the user by signing up or logging in. Use the following credentials for testing:
```txt
credentials:
    username: admin | user
    password: userpass
```
2. Upload an image in supported formats (JPG or PNG).
3. Click "Execute" to get the classification result.
4. The response will show the predicted class and confidence score.

## Docker Deployment
You can deploy the application using Docker. Follow these steps:

- You can pull the Docker image from Docker Hub:
```bash
docker pull jnlandu/api:latest
```
- Or, if you have cloned the repository, you can build the Docker image:
```bash
docker build -t your-preferred-image-name .
```
- To run the Docker container:
```bash
docker run -d -p 8000:8000 your-preferred-image-name
```
- Set the required environment variables in the `.env` file:
```bash
docker run -d -p 8000:8000 --env-file ./.env your-preferred-image-name
```
The application will be accessible at `http://localhost:8000`.

## Deployment with Azure
- The project can be deployed on Azure and accessed through the provided URL (if deployed).
- CI/CD pipelines are implemented using GitHub Actions. The workflow file is in the `.github/workflows` folder.

## Running Tests
The project includes a basic testing structure in the `/tests` folder. To run the tests:
```bash
pytest tests/
```
Authentication tests are available, and tests for image classification are in progress.

## In Progress
Here are features that are under development:
- **Frontend**: A user-friendly interface for the web app.
- **Model Improvements**: Support for additional image classification models.
- The mockup for the frontend is in the `mockup` folder, and you can preview the frontend design using Penpot [here](https://design.penpot.app/).

## Tech Stack
### Backend:
- **Python**
- **FastAPI**
- **Docker**
- **PyTorch** for deep learning model integration.
- **Azure Blob Storage** for storing uploaded images.
- **Azure PostgreSQL** for user authentication and session management.

### Frontend (In Progress):
- **HTML**
- **CSS**
- **Next.js 14**
- **Tailwind CSS**
- **TypeScript**
- **Shadcn and zod** for form validation.

## Authors
- [Jeremy N. Mabiala](https://jnlandu.github.io/)
- [Atou]()
- [Senanou]()

## Contributing
Contributions are welcome! Feel free to open issues for bug reports, feature requests, or questions.

## Acknowledgements
- [PyTorch](https://pytorch.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Azure Documentation](https://docs.microsoft.com/en-us/azure/)
- [Next.js Documentation](https://nextjs.org/docs)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
