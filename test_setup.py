"""
Test if CrabNet can be imported and list available models
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from crabnet.kingcrab import CrabNet
    from crabnet.model import Model
    print('✓ CrabNet modules imported successfully')
    
    # List available models
    models_dir = 'models/trained_models'
    if os.path.exists(models_dir):
        model_files = [f.replace('.pth', '') for f in os.listdir(models_dir) 
                       if f.endswith('.pth')]
        print(f'\n✓ Found {len(model_files)} pre-trained models')
        print('\nFirst 10 models:')
        for model in sorted(model_files)[:10]:
            print(f'  • {model}')
    else:
        print('⚠ Models directory not found')
        
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
