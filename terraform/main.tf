terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "5.73.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
  
}

# VPC
resource "aws_vpc" "ml_vpc" {
  cidr_block           = var.vpc_cidr_block
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "ml_vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "ml_internet_gateway" {
  vpc_id = aws_vpc.ml_vpc.id
  tags = {
    Name = "ml_internet_gateway"
  }
}

# Subnets
resource "aws_subnet" "ml_public_subnet" {
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = var.subnet_cidr_blocks[0]
  availability_zone = var.app_availability_zone

  tags = {
    Name = "ml_public_subnet"
  }
}

resource "aws_subnet" "ml_private_subnet_app" {
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = var.subnet_cidr_blocks[1]
  availability_zone = var.app_availability_zone

  tags = {
    Name = "ml_private_subnet_app"
  }
}

resource "aws_subnet" "ml_private_subnet_training" {
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = var.subnet_cidr_blocks[2]
  availability_zone = var.ml_availability_zone

  tags = {
    Name = "ml_private_subnet_training"
  }
}

# Elastic IP for NAT Gateway
resource "aws_eip" "nat_eip" {
  domain = "vpc"
  tags = {
    Name = "ml_nat_gateway_eip"
  }
}

# NAT Gateway
resource "aws_nat_gateway" "ml_nat_gateway" {
  allocation_id = aws_eip.nat_eip.id
  subnet_id     = aws_subnet.ml_public_subnet.id

  tags = {
    Name = "ml_nat_gateway"
  }

  depends_on = [aws_internet_gateway.ml_internet_gateway]
}

# Route Tables
resource "aws_route_table" "ml_public" {
  vpc_id = aws_vpc.ml_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.ml_internet_gateway.id
  }

  tags = { Name = "ml_public" }
}

resource "aws_route_table" "ml_private" {
  vpc_id = aws_vpc.ml_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.ml_nat_gateway.id
  }

  tags = { Name = "ml_private" }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.ml_public_subnet.id
  route_table_id = aws_route_table.ml_public.id
}

resource "aws_route_table_association" "private_app" {
  subnet_id      = aws_subnet.ml_private_subnet_app.id
  route_table_id = aws_route_table.ml_private.id
}

resource "aws_route_table_association" "private_training" {
  subnet_id      = aws_subnet.ml_private_subnet_training.id
  route_table_id = aws_route_table.ml_private.id
}

# Security Groups
resource "aws_security_group" "ml_frontend_security_group" {
  name_prefix = "ml_frontend_"
  vpc_id      = aws_vpc.ml_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP access"
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS access"
  }

  # Allow communication between frontend servers
  ingress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    self            = true
    description     = "Allow all traffic between frontend servers"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ml_frontend_security_group"
  }
}

resource "aws_security_group" "ml_monitoring_security_group" {
  name_prefix = "ml_monitoring_"
  vpc_id      = aws_vpc.ml_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }

  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Prometheus"
  }

  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Grafana"
  }


  # Allow communication between frontend servers
  ingress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    self            = true
    description     = "Allow all traffic between frontend servers"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ml_monitoring_security_group"
  }
}

resource "aws_security_group" "ml_backend_security_group" {
  vpc_id = aws_vpc.ml_vpc.id
  name   = "ML Backend Security"

  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.ml_frontend_security_group.id]
    description     = "SSH access from frontend"
  }

  ingress {
    description = "Allow inbound traffic on Node Exporters default port 9100"
    from_port   = 9100
    to_port     = 9100
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  
  }

  ingress {
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    security_groups = [aws_security_group.ml_frontend_security_group.id]
    description     = "API access from frontend"
  }
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description     = "Redis access from training server"
  }

  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
    self      = true  # This allows instances in this security group to communicate
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ml_backend_security_group"
  }
}

resource "aws_security_group" "ml_ui_security_group" {
  vpc_id = aws_vpc.ml_vpc.id
  name   = "ML UI Security"

  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.ml_frontend_security_group.id]
    description     = "SSH access from frontend"
  }

  ingress {
    from_port   = 5001
    to_port     = 5001
    protocol    = "tcp"
    security_groups = [aws_security_group.ml_frontend_security_group.id]
    description = "Gunicorn port for Flask UI"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ml_ui_security_group"
  }
}


# EC2 Instances Public subnet
resource "aws_instance" "ml_nginx_server" {
  ami                         = var.ec2_ami
  instance_type               = "t3.micro"
  subnet_id                   = aws_subnet.ml_public_subnet.id
  vpc_security_group_ids      = [aws_security_group.ml_frontend_security_group.id]
  key_name                    = var.key_name
  associate_public_ip_address = true
  depends_on = [aws_internet_gateway.ml_internet_gateway]
  user_data = <<-EOF
    #!/bin/bash
    # Redirect stdout and stderr to a log file
    exec > /var/log/user-data.log 2>&1
    echo "${file("./mykey.pem")}" > /home/ubuntu/.ssh/mykey.pem
    chmod 400 /home/ubuntu/.ssh/mykey.pem
    sudo chown ubuntu:ubuntu /home/ubuntu/.ssh/mykey.pem
    apt-get update -y
    apt-get install -y python3-pip
    apt-get install -y nginx
  EOF

  tags = {
    Name = "ml_nginx_server"
  }
}

resource "aws_instance" "ml_monitoring_server" {
  ami                         = var.ec2_ami
  instance_type               = "t3.micro"
  subnet_id                   = aws_subnet.ml_public_subnet.id
  vpc_security_group_ids      = [aws_security_group.ml_monitoring_security_group.id]
  key_name                    = var.key_name
  associate_public_ip_address = true
  depends_on = [aws_internet_gateway.ml_internet_gateway]
  user_data = "${file("monitoring_server.sh")}"

  tags = {
    Name = "ml_monitoring_server"
  }
}

# EC2 Instances Private subnet 

resource "aws_instance" "ml_ui_server" {
  ami                         = var.ec2_ami
  instance_type               = "t3.micro"
  subnet_id                   = aws_subnet.ml_private_subnet_app.id
  vpc_security_group_ids      = [aws_security_group.ml_ui_security_group.id]
  key_name                    = var.key_name
  user_data = templatefile("ml_frontend_server.sh", {
    priv_mykey = var.priv_mykey

  })
  depends_on = [aws_nat_gateway.ml_nat_gateway]

  tags = {
    Name = "ml_ui_server"
  }
}


resource "aws_instance" "ml_app_server" {
  ami                    = var.ec2_ami
  instance_type          = "t3.medium"
  subnet_id              = aws_subnet.ml_private_subnet_app.id
  vpc_security_group_ids = [aws_security_group.ml_backend_security_group.id]
  key_name               = var.key_name
  user_data = "${file("ml_app_server.sh")}"
  depends_on = [aws_nat_gateway.ml_nat_gateway]

  tags = {
    Name = "ml_app_server"
  }
}

resource "aws_instance" "ml_training_server" {
  ami                    = var.ec2_ami
  instance_type          = "p3.2xlarge"
  subnet_id              = aws_subnet.ml_private_subnet_training.id
  vpc_security_group_ids = [aws_security_group.ml_backend_security_group.id]
  key_name               = var.key_name
  user_data = "${file("ml_model_server.sh")}"
  depends_on = [aws_nat_gateway.ml_nat_gateway]

  root_block_device {
    volume_type           = "gp3"  # Standard General Purpose SSD
    volume_size           = 35    
    delete_on_termination = true
  }

  tags = {
    Name = "ml_training_server"
  }
}

output "nginx_ip" {
  value = aws_instance.ml_nginx_server.public_ip
}

output "monitoring_ip" {
  value = aws_instance.ml_monitoring_server.private_ip
}

output "ui_server_ip" {
  value = aws_instance.ml_ui_server.private_ip
}

output "nat_gateway_ip" {
  value = aws_eip.nat_eip.public_ip
}

output "ml_app_server_ip" {
  value = aws_instance.ml_app_server.private_ip
}

output "ml_model_server_ip" {
  value = aws_instance.ml_training_server.private_ip
}