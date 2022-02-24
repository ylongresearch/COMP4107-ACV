{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Practice_Generator.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyMm+tYTKp/v4VItzGDSDz1o",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ylongresearch/COMP4107-ACV/blob/main/Generator.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jBCNmhR8LP_t",
        "outputId": "268415e2-3ded-4810-e7ed-accf6bb00b84"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "torch.Size([2, 3, 256, 256])\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "\n",
        "class ConvBlock(nn.Module):\n",
        "    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):\n",
        "        super().__init__()\n",
        "        self.conv = nn.Sequential(\n",
        "            nn.Conv2d(in_channels, out_channels, padding_mode=\"reflect\", **kwargs)\n",
        "            if down\n",
        "            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),\n",
        "            nn.InstanceNorm2d(out_channels),\n",
        "            nn.ReLU(inplace=True) if use_act else nn.Identity()\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.conv(x)\n",
        "\n",
        "class ResidualBlock(nn.Module):\n",
        "    def __init__(self, channels):\n",
        "        super().__init__()\n",
        "        self.block = nn.Sequential(\n",
        "            ConvBlock(channels, channels, kernel_size=3, padding=1),\n",
        "            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        return x + self.block(x)\n",
        "\n",
        "class Generator(nn.Module):\n",
        "    def __init__(self, img_channels, num_features = 64, num_residuals=9):\n",
        "        super().__init__()\n",
        "        self.initial = nn.Sequential(\n",
        "            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode=\"reflect\"),\n",
        "            nn.InstanceNorm2d(num_features),\n",
        "            nn.ReLU(inplace=True),\n",
        "        )\n",
        "        self.down_blocks = nn.ModuleList(\n",
        "            [\n",
        "                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),\n",
        "                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),\n",
        "            ]\n",
        "        )\n",
        "        self.res_blocks = nn.Sequential(\n",
        "            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]\n",
        "        )\n",
        "        self.up_blocks = nn.ModuleList(\n",
        "            [\n",
        "                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),\n",
        "                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),\n",
        "            ]\n",
        "        )\n",
        "\n",
        "        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode=\"reflect\")\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.initial(x)\n",
        "        for layer in self.down_blocks:\n",
        "            x = layer(x)\n",
        "        x = self.res_blocks(x)\n",
        "        for layer in self.up_blocks:\n",
        "            x = layer(x)\n",
        "        return torch.tanh(self.last(x))\n",
        "\n",
        "def test():\n",
        "    img_channels = 3\n",
        "    img_size = 256\n",
        "    x = torch.randn((2, img_channels, img_size, img_size))\n",
        "    gen = Generator(img_channels, 9)\n",
        "    print(gen(x).shape)\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    test()"
      ]
    }
  ]
}