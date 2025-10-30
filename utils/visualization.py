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
    model.eval()
    device = next(model.parameters()).device
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    # === 1. RESUMEN TXT ===
    txt_path = save_path / "summary.txt"
    with open(txt_path, 'w') as f:
        print(f"ARQUITECTURA: {model_name.upper()}", file=f)
        print(f"Input: {input_size} | Device: {device}", file=f)
        print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}", file=f)
        print(f"Entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", file=f)
        print("="*80, file=f)
        try:
            summary(model, input_size=input_size, device=str(device).split(':')[0], file=f)
        except:
            print(model, file=f)
    logger.info(f"Resumen: {txt_path}")

    # === 2. GRAFO CON HIDDENLAYER (SOPORTA 3D) ===
    try:
        import hiddenlayer as hl

        # Crear tensor de entrada (batched)
        x = torch.randn(1, *input_size).to(device)  # [B, C, D, H, W]

        # Generar grafo
        graph = hl.build_graph(model, x)

        # Guardar PNG y SVG
        png_path = save_path / "graph.png"
        svg_path = save_path / "graph.svg"

        graph.save(str(png_path), format="png")
        graph.save(str(svg_path), format="svg")

        logger.info(f"Grafo 3D generado: {png_path} y {svg_path}")
    except Exception as e:
        logger.warning(f"hiddenlayer falló: {e}. Solo se generó summary.txt")

    # === 3. HTML ===
    html_path = save_path / "report.html"
    graph_html = ""
    if (save_path / "graph.png").exists():
        graph_html = f'''
        <div class="img-container">
            <img src="graph.png" alt="Grafo 3D">
            <p><a href="graph.svg">Descargar SVG (vectorial)</a></p>
        </div>
        '''
    else:
        graph_html = "<p>Grafo no disponible.</p>"

    with open(html_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{model_name.upper()}</title>
            <meta charset="utf-8">
            <style>
                body {{font-family:Arial; margin:40px; background:#f9f9f9}}
                .container {{max-width:1000px; margin:auto; background:white; padding:30px; border-radius:10px; box-shadow:0 0 10px rgba(0,0,0,0.1)}}
                h1 {{color:#2c3e50;}}
                pre {{background:#f4f4f4; padding:15px; border-radius:5px; overflow-x:auto; font-size:0.9em}}
                .img-container {{text-align:center; margin:30px 0}}
                img {{max-width:100%; border:1px solid #ddd; border-radius:5px}}
                a {{color:#3498db; text-decoration:none}}
                .footer {{margin-top:50px; font-size:0.8em; color:#95a5a6; text-align:center}}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{model_name.upper()}</h1>
                <p><strong>Input:</strong> {input_size} | <strong>Parámetros:</strong> {sum(p.numel() for p in model.parameters()):,}</p>
                <h2>Resumen de Capas</h2>
                <pre>{open(txt_path).read()}</pre>
                <h2>Grafo Computacional (3D)</h2>
                {graph_html}
                <div class="footer">
                    Generado con <code>hiddenlayer</code> + <code>torchsummary</code>
                </div>
            </div>
        </body>
        </html>
        """)
    logger.info(f"Reporte HTML: {html_path}")