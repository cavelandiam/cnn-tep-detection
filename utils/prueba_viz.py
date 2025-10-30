#!/usr/bin/env python3
import sys

print("=== DIAGNÓSTICO DE VISUALIZACIÓN ===\n")

# Test 1: PyTorch
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch no disponible: {e}")
    sys.exit(1)

# Test 2: torchview
try:
    import torchview
    print(f"✅ torchview disponible")
except ImportError:
    print(f"❌ torchview NO disponible")
    print("   Instalar con: pip install torchview")

# Test 3: graphviz (Python)
try:
    import graphviz
    print(f"✅ graphviz (Python) disponible")
except ImportError:
    print(f"❌ graphviz (Python) NO disponible")
    print("   Instalar con: pip install graphviz")

# Test 4: graphviz (Sistema)
import subprocess
try:
    result = subprocess.run(['dot', '-V'], capture_output=True, text=True)
    print(f"✅ graphviz (Sistema): {result.stderr.split()[4]}")
except FileNotFoundError:
    print(f"❌ graphviz (Sistema) NO disponible")
    print("   Instalar con: sudo apt-get install graphviz")

# Test 5: Generar grafo simple
print("\n=== TEST DE GENERACIÓN ===")
try:
    from torchview import draw_graph
    
    model = torch.nn.Conv3d(1, 8, 3)
    x = torch.randn(1, 1, 32, 32, 32)
    
    graph = draw_graph(model, input_data=x)
    graph.visual_graph.render('test_output', format='png', cleanup=True)
    
    print("✅ Grafo generado exitosamente: test_output.png")
    
except Exception as e:
    print(f"❌ Error al generar grafo: {e}")
    import traceback
    print(traceback.format_exc())