import torch
from torchsummary import summary
from pathlib import Path
from utils import logger
from datetime import datetime

def plot_model_architecture(
    model,
    save_dir: str,
    model_name: str,
    input_size=(1, 64, 128, 128, 1)
):
    """
    Genera un reporte completo de arquitectura 3D:
    - summary.txt
    - graph.png + graph.svg (con hiddenlayer → soporta Conv3d)
    - report.html (con todo integrado)

    Args:
        model: Modelo PyTorch (ResNet3D)
        save_dir: Carpeta base (ej: "graphs/rsna")
        model_name: Nombre base (ej: "resnet3d_pretrained")
        input_size: (C, D, H, W, 1) → Channel first, Depth
    """
    model.eval()
    device = next(model.parameters()).device
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    # === 1. RESUMEN TEXTUAL CON torchsummary ===
    txt_path = save_path / "summary.txt"
    with open(txt_path, 'w') as f:
        print(f"ARQUITECTURA: {model_name.upper()}", file=f)
        print(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=f)
        print(f"Input: {input_size} | Device: {device}", file=f)
        print(f"Total parámetros: {sum(p.numel() for p in model.parameters()):,}", file=f)
        print(f"Entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", file=f)
        print("="*80, file=f)
        try:
            summary(model, input_size=input_size, device=str(device).split(':')[0], file=f)
            logger.info(f"Resumen guardado: {txt_path}")
        except Exception as e:
            logger.warning(f"torchsummary falló: {e}. Usando print(model)")
            print(model, file=f)

    # === 2. GRAFO VISUAL CON HIDDENLAYER (SOPORTA 3D) ===
    png_path = save_path / "graph.png"
    svg_path = save_path / "graph.svg"
    graph_generated = False

    try:
        import hiddenlayer as hl

        # Entrada batched: [B, C, D, H, W]
        x = torch.randn(1, *input_size).to(device)

        # Construir grafo
        graph = hl.build_graph(model, x)

        # Guardar PNG y SVG
        graph.save(str(png_path), format="png")
        graph.save(str(svg_path), format="svg")

        graph_generated = True
        logger.info(f"Grafo 3D generado: {png_path} y {svg_path}")

    except ImportError as e:
        if "IPython" in str(e):
            logger.warning("hiddenlayer requiere IPython. Instala: pip install ipython")
        else:
            logger.warning(f"hiddenlayer no disponible: {e}")
    except Exception as e:
        logger.warning(f"hiddenlayer falló: {e}. Asegúrate de tener graphviz instalado.")

    # === 3. REPORTE HTML PROFESIONAL ===
    html_path = save_path / "report.html"
    graph_html = ""
    if graph_generated:
        graph_html = f'''
        <div class="img-container">
            <img src="{png_path.name}" alt="Grafo 3D del modelo">
            <p><a href="{svg_path.name}">Descargar SVG (vectorial)</a></p>
        </div>
        '''
    else:
        graph_html = '<p>Grafo no disponible. Instala: <code>pip install hiddenlayer ipython graphviz</code></p>'

    with open(html_path, 'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{model_name.upper()} - Arquitectura</title>
    <style>
        body {{font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f8f9fa; color: #333;}}
        .container {{max-width: 1100px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);}}
        h1 {{color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;}}
        h2 {{color: #2980b9; margin-top: 30px;}}
        pre {{background: #f4f4f4; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 0.9em; border-left: 4px solid #3498db;}}
        .img-container {{text-align: center; margin: 30px 0;}}
        img {{max-width: 100%; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}}
        a {{color: #3498db; text-decoration: none; font-weight: 500;}}
        a:hover {{text-decoration: underline;}}
        .footer {{margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; font-size: 0.85em; color: #7f8c8d; text-align: center;}}
        .badge {{background: #3498db; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.8em; font-weight: bold;}}
    </style>
</head>
<body>
<div class="container">
    <h1>{model_name.upper()}</h1>
    <p>
        <strong>Input:</strong> <code>{input_size}</code> | 
        <strong>Parámetros:</strong> <span class="badge">{sum(p.numel() for p in model.parameters()):,}</span> | 
        <strong>Dispositivo:</strong> <code>{device}</code>
    </p>

    <h2>Resumen de Capas</h2>
    <pre>{Path(txt_path).read_text()}</pre>

    <h2>Grafo Computacional (3D)</h2>
    {graph_html}

    <div class="footer">
        Generado con <code>torchsummary</code> + <code>hiddenlayer</code> | 
        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</div>
</body>
</html>
        """.strip())
    logger.info(f"Reporte HTML generado: {html_path}")