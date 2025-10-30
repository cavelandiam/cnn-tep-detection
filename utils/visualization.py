import sys
import torch
import torch.onnx
from io import StringIO
from pathlib import Path
from datetime import datetime
from utils import logger

def plot_model_architecture(
    model,
    save_dir: str,
    model_name: str,
    input_size=(1, 64, 128, 128, 1)
):
    """
    Genera un reporte completo de arquitectura 3D:
    - summary.txt (resumen textual)
    - graph.png + graph.svg (grafo visual si disponible)
    - report.html (reporte integrado)

    Args:
        model: Modelo PyTorch (ResNet3D)
        save_dir: Carpeta base (ej: "graphs/rsna")
        model_name: Nombre base (ej: "resnet3d_pretrained")
        input_size: (C, D, H, W) → Channel first, Depth
    """
    model.eval()
    device = next(model.parameters()).device
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Calcular parámetros totales
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # === 1. RESUMEN TEXTUAL ===
    txt_path = save_path / "summary.txt"
    summary_text = generate_model_summary(model, input_size, device, model_name, 
                                         total_params, trainable_params)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    logger.info(f"✅ Resumen guardado: {txt_path}")

    # === 2. GRAFO VISUAL ===
    png_path = save_path / "graph.png"
    svg_path = save_path / "graph.svg"
    graph_generated = generate_model_graph(model, input_size, device, png_path, svg_path)

    # === 3. REPORTE HTML ===
    html_path = save_path / "report.html"
    generate_html_report(html_path, model_name, input_size, device, 
                        total_params, trainable_params, summary_text,
                        graph_generated, png_path, svg_path)
    
    logger.info(f"📄 Reporte HTML generado: {html_path}")


def generate_model_summary(model, input_size, device, model_name, total_params, trainable_params):
    """Genera el resumen textual del modelo usando torchinfo o alternativa"""
    summary_lines = []
    summary_lines.append(f"{'='*80}")
    summary_lines.append(f"ARQUITECTURA: {model_name.upper()}")
    summary_lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"{'='*80}")
    summary_lines.append(f"Input Shape: {input_size}")
    summary_lines.append(f"Device: {device}")
    summary_lines.append(f"Total de Parámetros: {total_params:,}")
    summary_lines.append(f"Parámetros Entrenables: {trainable_params:,}")
    summary_lines.append(f"Parámetros No Entrenables: {total_params - trainable_params:,}")
    summary_lines.append(f"{'='*80}\n")

    # Intentar usar torchinfo (más moderno y compatible)
    try:
        from torchinfo import summary as torchinfo_summary
        
        # Capturar la salida de torchinfo
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        torchinfo_summary(
            model, 
            input_size=(1, *input_size),  # Agregar batch dimension
            device=str(device).split(':')[0],
            verbose=0
        )
        
        sys.stdout = old_stdout
        summary_lines.append(captured_output.getvalue())
        logger.info("✅ Resumen generado con torchinfo")
        
    except ImportError:
        logger.warning("⚠️ torchinfo no disponible. Instalando alternativa...")
        try:
            # Alternativa: usar torchsummary con captura de stdout
            from torchsummary import summary as torch_summary
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            torch_summary(model, input_size=input_size, device=str(device).split(':')[0])
            
            sys.stdout = old_stdout
            summary_lines.append(captured_output.getvalue())
            logger.info("✅ Resumen generado con torchsummary")
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo generar resumen automático: {e}")
            # Fallback: listar capas manualmente
            summary_lines.append("\n=== ESTRUCTURA DE CAPAS ===\n")
            for name, module in model.named_modules():
                if name:  # Evitar el módulo raíz
                    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    if params > 0:
                        summary_lines.append(f"{name}: {module.__class__.__name__} ({params:,} params)")
    
    return "\n".join(summary_lines)


def generate_model_graph(model, input_size, device, png_path, svg_path):
    """Genera el grafo visual del modelo (PNG y SVG)"""
    
    # Asegurar que torch esté importado globalmente
    import torch
    import torch.onnx
    
    # Método 1: Intentar con torchview (más moderno y compatible)
    try:
        from torchview import draw_graph
        
        logger.info("Intentando generar grafo con torchview...")
        
        # Entrada con batch dimension
        x = torch.randn(1, *input_size).to(device)
        
        # Generar grafo con manejo de errores robusto
        model_graph = draw_graph(
            model, 
            input_data=x,
            expand_nested=True,
            depth=10,
            device=str(device)  # Pasar como string para evitar problemas
        )
        
        # Guardar visualizaciones (sin extensión porque render() la agrega)
        base_path_png = str(png_path.parent / png_path.stem)
        base_path_svg = str(svg_path.parent / svg_path.stem)
        
        model_graph.visual_graph.render(base_path_png, format='png', cleanup=True)
        model_graph.visual_graph.render(base_path_svg, format='svg', cleanup=True)
        
        logger.info(f"✅ Grafo 3D generado con torchview: {png_path.name} y {svg_path.name}")
        return True
        
    except ImportError as e:
        logger.info(f"💡 torchview no disponible: {e}")
    except Exception as e:
        logger.warning(f"⚠️ torchview falló: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    # Método 2: Exportar a ONNX y usar netron
    try:
        logger.info("Intentando exportar modelo a ONNX...")
        
        # Verificar que onnx esté instalado
        try:
            import onnx
        except ImportError:
            logger.warning("⚠️ ONNX no instalado. Ejecuta: pip install onnx")
            raise
        
        # Exportar modelo a ONNX
        x = torch.randn(1, *input_size).to(device)
        onnx_path = png_path.parent / "model.onnx"
        
        # Mover modelo a CPU para exportación (más estable)
        model_cpu = model.cpu()
        x_cpu = x.cpu()
        
        torch.onnx.export(
            model_cpu,
            x_cpu,
            str(onnx_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11,
            do_constant_folding=True
        )
        
        # Volver modelo a dispositivo original
        model.to(device)
        
        logger.info(f"✅ Modelo exportado a ONNX: {onnx_path}")
        logger.info(f"💡 Visualiza con: netron {onnx_path}")
        logger.info(f"   O en el navegador: https://netron.app")
        
        # Crear imagen placeholder informativa
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            img = Image.new('RGB', (1000, 600), color='#f8f9fa')
            draw = ImageDraw.Draw(img)
            
            # Dibujar borde
            draw.rectangle([(20, 20), (980, 580)], outline='#667eea', width=3)
            
            # Título
            title = "Modelo 3D-CNN exportado a ONNX"
            
            # Texto principal
            text_lines = [
                "",
                f"Archivo: {onnx_path.name}",
                "",
                "Para visualizar el modelo:",
                "",
                "Opción 1 - Netron (local):",
                "  $ pip install netron",
                f"  $ netron {onnx_path.name}",
                "",
                "Opción 2 - Netron (web):",
                "  1. Visita: https://netron.app",
                f"  2. Carga el archivo: {onnx_path}",
                "",
                "El archivo ONNX contiene la arquitectura completa",
                "con todas las capas y conexiones del modelo 3D.",
            ]
            
            # Intentar usar fuente, si falla usar default
            try:
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
                font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                font_title = ImageFont.load_default()
                font_text = ImageFont.load_default()
            
            # Dibujar título
            draw.text((80, 60), title, fill='#667eea', font=font_title)
            
            # Dibujar texto
            y_offset = 130
            for line in text_lines:
                draw.text((80, y_offset), line, fill='#333333', font=font_text)
                y_offset += 28
            
            img.save(str(png_path))
            
            logger.info(f"✅ Imagen informativa guardada: {png_path.name}")
            logger.info(f"💡 El grafo real está en: {onnx_path.name}")
            
        except Exception as e:
            logger.warning(f"No se pudo crear imagen placeholder: {e}")
        
        return True  # ONNX exportado exitosamente
        
    except Exception as e:
        logger.warning(f"⚠️ Export ONNX falló: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    # Mensaje final de ayuda
    logger.info("💡 Para generar grafos visuales automáticamente:")
    logger.info("   1. pip install torchview onnx pillow")
    logger.info("   2. pip install graphviz")
    logger.info("   3. sudo apt-get install graphviz")
    logger.info("   4. Ejecuta de nuevo el script")
    
    return False


def generate_html_report(html_path, model_name, input_size, device, 
                        total_params, trainable_params, summary_text,
                        graph_generated, png_path, svg_path):
    """Genera el reporte HTML completo"""
    
    # Sección de gráficos
    if graph_generated:
        graph_html = f'''
        <div class="img-container">
            <img src="{png_path.name}" alt="Grafo 3D del modelo">
            <p><a href="{svg_path.name}" target="_blank">📥 Descargar SVG (vectorial)</a></p>
        </div>
        '''
    else:
        graph_html = '''
        <div class="warning-box">
            <p><strong>⚠️ Grafo no disponible</strong></p>
            <p>Para generar visualizaciones gráficas, instala:</p>
            <pre>pip install hiddenlayer ipython graphviz</pre>
            <p>En Ubuntu/Debian también necesitas: <code>sudo apt-get install graphviz</code></p>
        </div>
        '''
    
    # Escapar el summary_text para HTML
    summary_html = summary_text.replace('<', '&lt;').replace('>', '&gt;')
    
    html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name.upper()} - Arquitectura del Modelo</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: auto;
            background: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 25px;
            font-size: 2.2em;
        }}
        h2 {{
            color: #667eea;
            margin: 40px 0 20px 0;
            font-size: 1.8em;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .info-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 12px;
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        .info-card h3 {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 8px;
        }}
        .info-card p {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        pre {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.85em;
            border-left: 4px solid #667eea;
            line-height: 1.5;
        }}
        .img-container {{
            text-align: center;
            margin: 40px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        img:hover {{
            transform: scale(1.02);
        }}
        a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }}
        a:hover {{
            color: #764ba2;
            text-decoration: underline;
        }}
        .warning-box {{
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .warning-box p {{
            margin: 10px 0;
        }}
        .footer {{
            margin-top: 60px;
            padding-top: 30px;
            border-top: 2px solid #eee;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .badge {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            display: inline-block;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            color: #e83e8c;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>🧠 {model_name.upper()}</h1>
    
    <div class="info-grid">
        <div class="info-card">
            <h3>📐 Input Shape</h3>
            <p>{input_size}</p>
        </div>
        <div class="info-card">
            <h3>🔧 Parámetros Totales</h3>
            <p>{total_params:,}</p>
        </div>
        <div class="info-card">
            <h3>✅ Parámetros Entrenables</h3>
            <p>{trainable_params:,}</p>
        </div>
        <div class="info-card">
            <h3>💻 Dispositivo</h3>
            <p>{device}</p>
        </div>
    </div>

    <h2>📋 Resumen de Capas</h2>
    <pre>{summary_html}</pre>

    <h2>🔍 Grafo Computacional (3D)</h2>
    {graph_html}

    <div class="footer">
        <p>Generado con <code>PyTorch</code> + <code>torchinfo/torchsummary</code> + <code>hiddenlayer</code></p>
        <p><strong>Fecha:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</div>
</body>
</html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content.strip())