# src/llava_integration/attention_extraction.py
"""
LLaVA Integration для извлечения ground truth attention patterns
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import json

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    LLAVA_AVAILABLE = True
except ImportError:
    print("⚠️  LLaVA not installed. Using mock implementation.")
    LLAVA_AVAILABLE = False


class LLaVAAttentionExtractor:
    """Извлечение attention patterns из LLaVA модели"""
    
    def __init__(self, model_path: str = "liuhaotian/llava-v1.5-7b", device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        
        if LLAVA_AVAILABLE:
            self._load_llava_model()
        else:
            self._create_mock_model()
    
    def _load_llava_model(self):
        """Загружает настоящую LLaVA модель"""
        try:
            model_name = get_model_name_from_path(self.model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                self.model_path, None, model_name, device_map=self.device
            )
            self.model.eval()
            print(f"✅ LLaVA model loaded: {model_name}")
            
        except Exception as e:
            print(f"❌ Failed to load LLaVA: {e}")
            self._create_mock_model()
    
    def _create_mock_model(self):
        """Создает mock модель для тестирования"""
        print("🔧 Using mock LLaVA implementation")
        
        # Mock tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Mock model components
        self.model = None
        self.image_processor = None
        self.context_len = 2048
    
    def extract_attention_weights(
        self, 
        image: Image.Image, 
        question: str,
        return_layers: List[int] = [-1]  # Последний слой по умолчанию
    ) -> Dict[str, torch.Tensor]:
        """
        Извлекает attention weights между visual и text tokens
        
        Args:
            image: PIL Image
            question: Text question
            return_layers: Слои для извлечения attention
            
        Returns:
            Dictionary с attention weights и metadata
        """
        
        if not LLAVA_AVAILABLE or self.model is None:
            return self._mock_attention_extraction(image, question)
        
        try:
            return self._real_attention_extraction(image, question, return_layers)
        except Exception as e:
            print(f"⚠️  Attention extraction failed: {e}")
            return self._mock_attention_extraction(image, question)
    
    def _real_attention_extraction(
        self, 
        image: Image.Image, 
        question: str,
        return_layers: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Настоящее извлечение attention из LLaVA"""
        
        # Подготовка conversation template
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Токенизация
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        # Обработка изображения
        image_tensor = self.image_processor.preprocess(
            image, return_tensors='pt'
        )['pixel_values'][0].unsqueeze(0).to(self.device)
        
        # Hook для перехвата attention weights
        attention_weights = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                # output[1] содержит attention weights для MultiHeadAttention
                if len(output) > 1 and output[1] is not None:
                    attention_weights[name] = output[1].detach().cpu()
            return hook
        
        # Регистрируем hooks на нужных слоях
        handles = []
        for layer_idx in return_layers:
            if layer_idx < 0:
                layer_idx = len(self.model.model.layers) + layer_idx
            
            layer_name = f"layer_{layer_idx}"
            layer = self.model.model.layers[layer_idx]
            
            # Hook на self-attention
            handle = layer.self_attn.register_forward_hook(attention_hook(layer_name))
            handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=1,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        # Удаляем hooks
        for handle in handles:
            handle.remove()
        
        # Обрабатываем attention weights
        processed_attention = self._process_attention_weights(
            attention_weights, input_ids, len(self.tokenizer.encode(question))
        )
        
        return processed_attention
    
    def _mock_attention_extraction(
        self, 
        image: Image.Image, 
        question: str
    ) -> Dict[str, torch.Tensor]:
        """Mock implementation для тестирования"""
        
        # Генерируем реалистичные fake attention patterns
        num_visual_tokens = 576  # 24x24 patches
        question_tokens = len(question.split()) + 5  # Приблизительно
        
        # Создаем attention pattern с некоторой логикой
        attention_pattern = torch.zeros(num_visual_tokens)
        
        # Симулируем фокус в зависимости от типа вопроса
        if "color" in question.lower():
            # Фокус на центральных patches
            center_patches = torch.arange(200, 376)  # Центральная область
            attention_pattern[center_patches] = torch.rand(len(center_patches)) * 0.8 + 0.2
        
        elif "how many" in question.lower():
            # Распределенное внимание
            attention_pattern = torch.rand(num_visual_tokens) * 0.6 + 0.1
        
        elif "where" in question.lower():
            # Фокус на краях изображения
            edge_patches = list(range(0, 100)) + list(range(476, 576))
            attention_pattern[edge_patches] = torch.rand(len(edge_patches)) * 0.7 + 0.2
        
        else:
            # Общий паттерн
            attention_pattern = torch.rand(num_visual_tokens) * 0.5 + 0.1
        
        # Нормализация
        attention_pattern = torch.softmax(attention_pattern, dim=0)
        
        return {
            'visual_attention': attention_pattern,
            'attention_scores': attention_pattern,
            'num_visual_tokens': num_visual_tokens,
            'num_text_tokens': question_tokens,
            'question': question,
            'is_mock': True
        }
    
    def _process_attention_weights(
        self, 
        raw_attention: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        question_length: int
    ) -> Dict[str, torch.Tensor]:
        """Обрабатывает raw attention weights"""
        
        processed = {}
        
        for layer_name, attention in raw_attention.items():
            # attention shape: [batch, num_heads, seq_len, seq_len]
            batch_size, num_heads, seq_len, _ = attention.shape
            
            # Усредняем по heads
            avg_attention = attention.mean(dim=1)  # [batch, seq_len, seq_len]
            
            # Определяем границы visual и text tokens
            # Это зависит от конкретной реализации LLaVA
            # Обычно: [system_tokens] + [visual_tokens] + [question_tokens]
            
            # Приблизительное разделение (нужно адаптировать под конкретную LLaVA)
            visual_start = 5  # После system tokens
            visual_end = visual_start + 576  # 576 visual tokens
            text_start = visual_end
            text_end = text_start + question_length
            
            if visual_end < seq_len and text_end <= seq_len:
                # Извлекаем attention от visual tokens к text tokens
                visual_to_text = avg_attention[0, visual_start:visual_end, text_start:text_end]
                
                # Усредняем по text tokens для получения важности каждого visual token
                visual_importance = visual_to_text.mean(dim=1)  # [num_visual_tokens]
                
                processed[layer_name] = {
                    'visual_attention': visual_importance,
                    'visual_to_text': visual_to_text,
                    'full_attention': avg_attention[0]
                }
        
        # Объединяем результаты разных слоев
        if processed:
            # Используем последний слой как основной
            main_layer = list(processed.keys())[-1]
            main_attention = processed[main_layer]['visual_attention']
            
            return {
                'visual_attention': main_attention,
                'attention_scores': main_attention,
                'num_visual_tokens': len(main_attention),
                'layer_attention': processed,
                'is_mock': False
            }
        else:
            # Fallback к mock если что-то пошло не так
            return self._mock_attention_extraction(None, "")


class VQAGroundTruthGenerator:
    """Генератор ground truth данных для VQA fine-tuning"""
    
    def __init__(self, llava_model_path: str = "liuhaotian/llava-v1.5-7b", device: str = "cuda"):
        self.attention_extractor = LLaVAAttentionExtractor(llava_model_path, device)
        self.device = device
    
    def generate_vqa_ground_truth(
        self,
        image_path: str,
        question: str,
        answer: str = None
    ) -> Dict:
        """
        Генерирует ground truth данные для одного VQA примера
        
        Returns:
            Dictionary с visual features, question embeddings, target attention
        """
        
        # Загрузка изображения
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
        
        # Извлечение attention patterns
        attention_data = self.attention_extractor.extract_attention_weights(image, question)
        
        # Извлечение visual features (как в COCO dataset)
        from ..data.coco_pretraining import FeatureExtractor
        from ..config.base_config import ExperimentConfig
        
        config = ExperimentConfig()
        feature_extractor = FeatureExtractor(config)
        
        try:
            visual_features = feature_extractor.extract_visual_features(image)
            question_embeds = feature_extractor.extract_text_features(question)
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
        
        return {
            'visual_features': visual_features,
            'question_embeds': question_embeds,
            'target_attention': attention_data['visual_attention'],
            'attention_scores': attention_data['attention_scores'],
            'question': question,
            'answer': answer,
            'image_path': image_path,
            'is_mock_attention': attention_data.get('is_mock', False)
        }
    
    def process_vqa_dataset(
        self,
        vqa_questions_file: str,
        vqa_annotations_file: str,
        coco_images_dir: str,
        output_file: str,
        max_samples: Optional[int] = None
    ):
        """
        Обрабатывает весь VQA dataset и создает ground truth файл
        """
        
        print(f"🔄 Processing VQA dataset...")
        print(f"Questions: {vqa_questions_file}")
        print(f"Annotations: {vqa_annotations_file}")
        print(f"Images: {coco_images_dir}")
        
        # Загружаем VQA данные
        with open(vqa_questions_file, 'r') as f:
            questions_data = json.load(f)
        
        if vqa_annotations_file:
            with open(vqa_annotations_file, 'r') as f:
                annotations_data = json.load(f)
            
            # Создаем mapping question_id -> answer
            id_to_answer = {}
            for ann in annotations_data['annotations']:
                id_to_answer[ann['question_id']] = ann['multiple_choice_answer']
        else:
            id_to_answer = {}
        
        # Обрабатываем samples
        processed_samples = []
        questions = questions_data['questions']
        
        if max_samples:
            questions = questions[:max_samples]
        
        for i, q_data in enumerate(questions):
            if i % 100 == 0:
                print(f"Processed {i}/{len(questions)} samples...")
            
            question_id = q_data['question_id']
            question = q_data['question']
            image_id = q_data['image_id']
            
            # Формируем путь к изображению
            image_filename = f"COCO_val2014_{image_id:012d}.jpg"
            image_path = os.path.join(coco_images_dir, image_filename)
            
            if not os.path.exists(image_path):
                # Попробуем другой формат
                image_filename = f"{image_id:012d}.jpg"
                image_path = os.path.join(coco_images_dir, image_filename)
            
            if not os.path.exists(image_path):
                continue
            
            # Получаем answer
            answer = id_to_answer.get(question_id, "")
            
            # Генерируем ground truth
            gt_data = self.generate_vqa_ground_truth(image_path, question, answer)
            
            if gt_data:
                gt_data['question_id'] = question_id
                gt_data['image_id'] = image_id
                processed_samples.append(gt_data)
        
        # Сохраняем результат
        print(f"💾 Saving {len(processed_samples)} processed samples to {output_file}")
        torch.save(processed_samples, output_file)
        
        print(f"✅ VQA ground truth generation completed!")
        return processed_samples


# Test и utility functions
def test_attention_extraction():
    """Тест извлечения attention"""
    
    extractor = LLaVAAttentionExtractor(device="cpu")
    
    # Создаем test image
    test_image = Image.new('RGB', (336, 336), color='red')
    test_question = "What color is this image?"
    
    # Извлекаем attention
    attention_data = extractor.extract_attention_weights(test_image, test_question)
    
    print("✅ Attention extraction test:")
    print(f"  Visual attention shape: {attention_data['visual_attention'].shape}")
    print(f"  Attention sum: {attention_data['visual_attention'].sum():.4f}")
    print(f"  Max attention: {attention_data['visual_attention'].max():.4f}")
    print(f"  Is mock: {attention_data.get('is_mock', False)}")
    
    return attention_data


if __name__ == "__main__":
    # Тест
    print("🧪 Testing LLaVA attention extraction...")
    test_attention_extraction()
    print("✅ Test completed!")
