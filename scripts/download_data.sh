#!/bin/bash
"""
Data Download Script для COCO и VQA datasets
"""

echo "📁 Setting up datasets for Question-Conditioned LLaVA..."

# Create data directories
mkdir -p data/{coco,vqav2,cache}

echo ""
echo "🔽 COCO Dataset Download"
echo "========================"

# COCO Images
echo "Downloading COCO 2017 train images..."
cd data/coco
if [ ! -f "train2017.zip" ]; then
    wget http://images.cocodataset.org/zips/train2017.zip
    echo "✅ Downloaded train2017.zip"
else
    echo "⏭️  train2017.zip already exists"
fi

echo "Downloading COCO 2017 val images..."
if [ ! -f "val2017.zip" ]; then
    wget http://images.cocodataset.org/zips/val2017.zip
    echo "✅ Downloaded val2017.zip"
else
    echo "⏭️  val2017.zip already exists"
fi

# COCO Annotations
echo "Downloading COCO annotations..."
if [ ! -f "annotations_trainval2017.zip" ]; then
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    echo "✅ Downloaded annotations_trainval2017.zip"
else
    echo "⏭️  annotations_trainval2017.zip already exists"
fi

# Extract COCO files
echo "Extracting COCO files..."
if [ ! -d "train2017" ]; then
    unzip -q train2017.zip
    echo "✅ Extracted train2017/"
fi

if [ ! -d "val2017" ]; then
    unzip -q val2017.zip  
    echo "✅ Extracted val2017/"
fi

if [ ! -d "annotations" ]; then
    unzip -q annotations_trainval2017.zip
    echo "✅ Extracted annotations/"
fi

echo ""
echo "🔽 VQAv2 Dataset Download" 
echo "========================="

cd ../vqav2

# VQA Questions
echo "Downloading VQA questions..."
if [ ! -f "v2_Questions_Train_mscoco.zip" ]; then
    wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
    echo "✅ Downloaded training questions"
else
    echo "⏭️  Training questions already exist"
fi

if [ ! -f "v2_Questions_Val_mscoco.zip" ]; then
    wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
    echo "✅ Downloaded validation questions"
else
    echo "⏭️  Validation questions already exist"
fi

# VQA Annotations
echo "Downloading VQA annotations..."
if [ ! -f "v2_Annotations_Train_mscoco.zip" ]; then
    wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
    echo "✅ Downloaded training annotations"
else
    echo "⏭️  Training annotations already exist"
fi

if [ ! -f "v2_Annotations_Val_mscoco.zip" ]; then
    wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
    echo "✅ Downloaded validation annotations"
else
    echo "⏭️  Validation annotations already exist"
fi

# Extract VQA files
echo "Extracting VQA files..."
unzip -q -o "*.zip"
echo "✅ VQA files extracted"

cd ../..

echo ""
echo "📊 Dataset Summary"
echo "=================="
echo "COCO Train Images: $(find data/coco/train2017 -name "*.jpg" | wc -l 2>/dev/null || echo "0") files"
echo "COCO Val Images: $(find data/coco/val2017 -name "*.jpg" | wc -l 2>/dev/null || echo "0") files"
echo "VQA Files: $(find data/vqav2 -name "*.json" | wc -l 2>/dev/null || echo "0") files"

echo ""
echo "✅ Dataset download completed!"
echo ""
echo "🎯 Next steps:"
echo "1. Test the setup: python scripts/run_pretraining.py --config_variant debug"
echo "2. Start COCO pretraining: python scripts/run_pretraining.py"
echo "3. Fine-tune on VQA: python scripts/run_finetuning.py --pretrained_selector outputs/pretrained_selector.pt"
