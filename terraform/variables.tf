variable "vpc_cidr_block" {
  description = "CIDR block for the ML VPC"
  default     = "10.0.0.0/16"
}

variable "subnet_cidr_blocks" {
  description = "CIDR blocks for ML subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "ml_availability_zone" {
  description = "Availability Zone for ML infrastructure"
  default     = "us-east-1a"
}

variable "app_availability_zone" {
  description = "Availability Zone for ML infrastructure"
  default     = "us-east-1a"
}

variable "ec2_ami" {
  description = "AMI ID for ML EC2 instances"
  default     = "ami-0866a3c8686eaeeba"
}

variable "key_name" {
  description = "Key pair name for ML EC2 instances"
  default     = "mykey"
}



# variable "my_key_path" {
#   default = "./mykey.pem"
#   description = "Path to the private key file"
# }

variable "priv_mykey" {
  sensitive = true
}