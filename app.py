#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import streamlit as st
import os
from PIL import Image
import numpy as np


# In[2]:


# Define Generator class
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# In[3]:


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
z_dim = 100
G = Generator(z_dim).to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()


# In[4]:


# Streamlit Web App
st.title("Anime GAN Image Generator")
st.write("Enter the number of images to generate and click the button!")

num_images = st.text_input("Enter number of images to generate (1-10):", "5")

if st.button("Generate Images"):
    try:
        num_images = int(num_images)
        if num_images < 1 or num_images > 10:
            st.error("Please enter a number between 1 and 10.")
        else:
            noise = torch.randn(num_images, z_dim, 1, 1, device=device)
            with torch.no_grad():
                fake_images = G(noise).cpu()
            
            fake_images = fake_images.mul(0.5).add(0.5).clamp(0, 1)  # De-normalize
            fake_images = fake_images.numpy().transpose(0, 2, 3, 1)  # Convert to (batch, H, W, C)
            
            st.write("Generated Images:")
            for img in fake_images:
                img = (img * 255).astype(np.uint8)
                image_pil = Image.fromarray(img)
                st.image(image_pil)
    except ValueError:
        st.error("Invalid input. Please enter a valid integer between 1 and 10.")
