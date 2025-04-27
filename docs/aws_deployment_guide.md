# AWS Deployment Guide

## 1. Create EC2 Instance
- AWS Console → EC2 → Launch Instance
- Choose Ubuntu 22.04 LTS AMI
- Choose t2.medium or t2.large
- Configure Security Group:
  - Allow inbound ports: 22 (SSH), 8000 (API), 8501 (Streamlit)

## 2. SSH into Instance
```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

## 3. Install Docker
```bash
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

## 4. Clone Repo
```bash
git clone https://github.com/yourusername/rag-shopping-assistant.git
cd rag-shopping-assistant
```

## 5. Build and Run Docker
```bash
docker build -t rag-api .
docker run -d -p 8000:8000 rag-api
```

## 6. Run Streamlit App
```bash
streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
```

