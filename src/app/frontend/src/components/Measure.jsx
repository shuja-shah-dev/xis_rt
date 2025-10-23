import React, { useState, useRef, useEffect } from "react";
import { BlobLarge } from "./Blob";

const ImageMeasurementTool = () => {
  const [tool, setTool] = useState("bbox");

  const [imageLoaded, setImageLoaded] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [isMoving, setIsMoving] = useState(false);
  const [startPoint, setStartPoint] = useState(null);
  const [endPoint, setEndPoint] = useState(null);
  const [roi, setRoi] = useState(null);
  const [polygonPoints, setPolygonPoints] = useState([]);
  const [polygonClosed, setPolygonClosed] = useState(false);
  const [hoverPoint, setHoverPoint] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [lastPanPoint, setLastPanPoint] = useState(null);
  const [moveStartPoint, setMoveStartPoint] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });
  const [initialScale, setInitialScale] = useState(1);
  const [showResult, setShowResult] = useState(false);
  const [resultData, setResultData] = useState(null);

  const canvasRef = useRef(null);
  const imageRef = useRef(new Image());
  const containerRef = useRef(null);

   const [videoMode, setVideoMode] = useState(null);
    const imageUrl =
      videoMode === "baked_baguette"
        ? "baguette.png"
        : videoMode === "donut"
        ? "frame_1.png"
        : null;
  
    useEffect(() => {
      const fetchConfig = async () => {
        try {
          const res = await fetch("http://localhost:5000/api/config");
          const data = await res.json();
          if (res.ok) {
            setVideoMode(data.video_mode);
          }
        } catch (err) {
          console.error("Error fetching config:", err);
        }
      };
  
      fetchConfig();
    }, []);

  useEffect(() => {
    const img = imageRef.current;
    img.onload = () => {
      setImageLoaded(true);
      setImageSize({ width: img.width, height: img.height });
      fitImageToCanvas();
    };
    img.src = imageUrl;
  }, [imageUrl]);


  useEffect(() => {
    if (containerRef.current) {
      const resizeObserver = new ResizeObserver(() => {
        fitImageToCanvas();
      });
      resizeObserver.observe(containerRef.current);
      return () => resizeObserver.disconnect();
    }
  }, []);

  useEffect(() => {
    drawCanvas();
  }, [
    imageLoaded,
    roi,
    polygonPoints,
    polygonClosed,
    hoverPoint,
    startPoint,
    endPoint,
    zoom,
    pan,
  ]);

  const fitImageToCanvas = () => {
    if (!containerRef.current || !imageRef.current.width) return;

    const container = containerRef.current;
    const maxWidth = container.clientWidth;
    const maxHeight = window.innerHeight * 0.7;

    const imgWidth = imageRef.current.width;
    const imgHeight = imageRef.current.height;

    const scale = Math.min(maxWidth / imgWidth, maxHeight / imgHeight, 1);
    setInitialScale(scale);

    const newWidth = imgWidth * scale;
    const newHeight = imgHeight * scale;

    setCanvasSize({ width: newWidth, height: newHeight });
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !imageLoaded) return;

    const ctx = canvas.getContext("2d");
    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);
    ctx.translate(-canvas.width / 2, -canvas.height / 2);

    ctx.drawImage(
      imageRef.current,
      0,
      0,
      imageRef.current.width,
      imageRef.current.height,
      0,
      0,
      canvasSize.width,
      canvasSize.height
    );

    if (roi) {
      ctx.strokeStyle = "#4caf50";
      ctx.lineWidth = 2 / zoom;
      ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
      ctx.fillStyle = "rgba(76, 175, 80, 0.1)";
      ctx.fillRect(roi.x, roi.y, roi.width, roi.height);
    }

    if (isDrawing && startPoint && endPoint && tool === "bbox") {
      ctx.strokeStyle = "#4caf50";
      ctx.lineWidth = 2 / zoom;
      ctx.setLineDash([5 / zoom, 5 / zoom]);
      const width = endPoint.x - startPoint.x;
      const height = endPoint.y - startPoint.y;
      ctx.strokeRect(startPoint.x, startPoint.y, width, height);
      ctx.setLineDash([]);
    }

    if (polygonPoints.length > 0) {
      ctx.strokeStyle = "#2196f3";
      ctx.lineWidth = 2 / zoom;
      ctx.fillStyle = "rgba(33, 150, 243, 0.2)";

      ctx.beginPath();
      ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);

      for (let i = 1; i < polygonPoints.length; i++) {
        ctx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
      }

      if (polygonClosed) {
        ctx.closePath();
        ctx.fill();
      } else if (hoverPoint) {
        ctx.lineTo(hoverPoint.x, hoverPoint.y);
      }

      ctx.stroke();

      polygonPoints.forEach((point, index) => {
        ctx.fillStyle = "#f44336";
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2 / zoom;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5 / zoom, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        if (index === 0 && polygonPoints.length > 2 && !polygonClosed) {
          ctx.strokeStyle = "#ffc107";
          ctx.lineWidth = 2 / zoom;
          ctx.beginPath();
          ctx.arc(point.x, point.y, 10 / zoom, 0, 2 * Math.PI);
          ctx.stroke();
        }
      });
    }

    ctx.restore();
  };

  const screenToCanvas = (screenX, screenY) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const canvas = canvasRef.current;

    const canvasX = screenX - rect.left;
    const canvasY = screenY - rect.top;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    const x = (canvasX - centerX - pan.x) / zoom + centerX;
    const y = (canvasY - centerY - pan.y) / zoom + centerY;

    return { x, y };
  };

  const canvasToOriginalImage = (canvasX, canvasY) => {
    return {
      x: canvasX / initialScale,
      y: canvasY / initialScale,
    };
  };

  const isPointInRoi = (point, roi) => {
    return (
      point.x >= roi.x &&
      point.x <= roi.x + roi.width &&
      point.y >= roi.y &&
      point.y <= roi.y + roi.height
    );
  };

  const isPointInPolygon = (point, polygon) => {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i].x,
        yi = polygon[i].y;
      const xj = polygon[j].x,
        yj = polygon[j].y;

      const intersect =
        yi > point.y !== yj > point.y &&
        point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  };

  const isNearFirstPoint = (pos) => {
    if (polygonPoints.length < 3) return false;
    const first = polygonPoints[0];
    const distance = Math.sqrt(
      Math.pow(pos.x - first.x, 2) + Math.pow(pos.y - first.y, 2)
    );
    return distance < 10 / zoom;
  };

  const handleMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    const pos = screenToCanvas(e.clientX, e.clientY);

    if (tool === "pan") {
      setIsPanning(true);
      setLastPanPoint({ x: e.clientX, y: e.clientY });
    } else if (tool === "move") {
      if (roi && isPointInRoi(pos, roi)) {
        setIsMoving(true);
        setMoveStartPoint(pos);
      } else if (polygonClosed && isPointInPolygon(pos, polygonPoints)) {
        setIsMoving(true);
        setMoveStartPoint(pos);
      }
    } else if (tool === "bbox") {
      setStartPoint(pos);
      setEndPoint(pos);
      setIsDrawing(true);
    } else if (tool === "polygon" && !polygonClosed) {
      if (isNearFirstPoint(pos) && polygonPoints.length > 2) {
        setPolygonClosed(true);
      } else {
        setPolygonPoints([...polygonPoints, pos]);
      }
    }
  };

  const handleMouseMove = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const pos = screenToCanvas(e.clientX, e.clientY);

    if (isPanning && lastPanPoint) {
      const dx = e.clientX - lastPanPoint.x;
      const dy = e.clientY - lastPanPoint.y;
      setPan((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
      setLastPanPoint({ x: e.clientX, y: e.clientY });
    } else if (isMoving && moveStartPoint) {
      const dx = pos.x - moveStartPoint.x;
      const dy = pos.y - moveStartPoint.y;

      if (roi && tool === "move") {
        setRoi({
          ...roi,
          x: roi.x + dx,
          y: roi.y + dy,
        });
      } else if (polygonClosed && tool === "move") {
        setPolygonPoints(
          polygonPoints.map((p) => ({
            x: p.x + dx,
            y: p.y + dy,
          }))
        );
      }
      setMoveStartPoint(pos);
    } else if (isDrawing && tool === "bbox") {
      setEndPoint(pos);
    } else if (
      tool === "polygon" &&
      !polygonClosed &&
      polygonPoints.length > 0
    ) {
      setHoverPoint(pos);
    }
  };

  const handleMouseUp = () => {
    if (isPanning) {
      setIsPanning(false);
      setLastPanPoint(null);
    } else if (isMoving) {
      setIsMoving(false);
      setMoveStartPoint(null);
    } else if (isDrawing && tool === "bbox" && startPoint && endPoint) {
      const width = Math.abs(endPoint.x - startPoint.x);
      const height = Math.abs(endPoint.y - startPoint.y);
      const x = Math.min(startPoint.x, endPoint.x);
      const y = Math.min(startPoint.y, endPoint.y);

      if (width > 5 && height > 5) {
        setRoi({ x, y, width, height });
      }
      setIsDrawing(false);
      setStartPoint(null);
      setEndPoint(null);
    }
  };

  const handleMouseLeave = () => {
    setHoverPoint(null);
    handleMouseUp();
  };

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev * 1.2, 5));
  };

  const handleZoomOut = () => {
    if (zoom > 1) {
      setZoom((prev) => Math.max(prev / 1.2, 0.5));
    }
  };

  const handleReset = () => {
    setTool("bbox");
    setRoi(null);
    setPolygonPoints([]);
    setPolygonClosed(false);
    setHoverPoint(null);
    setStartPoint(null);
    setEndPoint(null);
    setIsMoving(false);
    setMoveStartPoint(null);
    fitImageToCanvas();
  };

  const handleErase = () => {
    if (tool === "bbox") {
      setRoi(null);
    } else if (tool === "polygon") {
      setPolygonPoints([]);
      setPolygonClosed(false);
      setHoverPoint(null);
    }
  };

  const handleUndo = () => {
    if (tool === "polygon") {
      if (polygonClosed) {
        setPolygonClosed(false);
      } else if (polygonPoints.length > 0) {
        setPolygonPoints(polygonPoints.slice(0, -1));
      }
    }
  };

  const handleExpandRegion = () => {
    if (roi && tool === "bbox") {
      setRoi({
        x: roi.x - 5,
        y: roi.y - 5,
        width: roi.width + 10,
        height: roi.height + 10,
      });
    } else if (polygonClosed && tool === "polygon") {
      const centerX =
        polygonPoints.reduce((sum, p) => sum + p.x, 0) / polygonPoints.length;
      const centerY =
        polygonPoints.reduce((sum, p) => sum + p.y, 0) / polygonPoints.length;

      setPolygonPoints(
        polygonPoints.map((p) => ({
          x: centerX + (p.x - centerX) * 1.1,
          y: centerY + (p.y - centerY) * 1.1,
        }))
      );
    }
  };

  const handleShrinkRegion = () => {
    if (roi && tool === "bbox" && roi.width > 20 && roi.height > 20) {
      setRoi({
        x: roi.x + 5,
        y: roi.y + 5,
        width: roi.width - 10,
        height: roi.height - 10,
      });
    } else if (polygonClosed && tool === "polygon") {
      const centerX =
        polygonPoints.reduce((sum, p) => sum + p.x, 0) / polygonPoints.length;
      const centerY =
        polygonPoints.reduce((sum, p) => sum + p.y, 0) / polygonPoints.length;

      setPolygonPoints(
        polygonPoints.map((p) => ({
          x: centerX + (p.x - centerX) * 0.9,
          y: centerY + (p.y - centerY) * 0.9,
        }))
      );
    }
  };

  const handleGenerate = () => {
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = imageRef.current.width;
    maskCanvas.height = imageRef.current.height;
    const maskCtx = maskCanvas.getContext("2d");

    maskCtx.fillStyle = "black";
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskCtx.fillStyle = "white";

    if (roi) {
      const originalRoi = canvasToOriginalImage(roi.x, roi.y);
      const originalWidth = roi.width / initialScale;
      const originalHeight = roi.height / initialScale;
      maskCtx.fillRect(
        originalRoi.x,
        originalRoi.y,
        originalWidth,
        originalHeight
      );

      const maskBase64 = maskCanvas.toDataURL("image/png").split(",")[1];

      setResultData({
        region: {
          type: "rectangle",
          x: Math.round(originalRoi.x),
          y: Math.round(originalRoi.y),
          width: Math.round(originalWidth),
          height: Math.round(originalHeight),
        },
        mask_base64: maskBase64,
      });
    } else if (polygonClosed && polygonPoints.length > 2) {
      maskCtx.beginPath();
      const originalPoints = polygonPoints.map((p) =>
        canvasToOriginalImage(p.x, p.y)
      );
      maskCtx.moveTo(originalPoints[0].x, originalPoints[0].y);

      for (let i = 1; i < originalPoints.length; i++) {
        maskCtx.lineTo(originalPoints[i].x, originalPoints[i].y);
      }

      maskCtx.closePath();
      maskCtx.fill();

      const maskBase64 = maskCanvas.toDataURL("image/png").split(",")[1];

      const xs = originalPoints.map((p) => p.x);
      const ys = originalPoints.map((p) => p.y);

      setResultData({
        region: {
          type: "polygon",
          vertices: originalPoints.map((p) => ({
            x: Math.round(p.x),
            y: Math.round(p.y),
          })),
          bounds: {
            x: Math.round(Math.min(...xs)),
            y: Math.round(Math.min(...ys)),
            width: Math.round(Math.max(...xs) - Math.min(...xs)),
            height: Math.round(Math.max(...ys) - Math.min(...ys)),
          },
        },
        mask_base64: maskBase64,
      });
    }

    setShowResult(true);
    console.log("Mask Data:", resultData);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageUrl(e.target.result);
        handleReset();
      };
      reader.readAsDataURL(file);
    }
  };

  const getCursorStyle = () => {
    if (tool === "pan") return "grab";
    if (tool === "move") return "move";
    if (tool === "bbox" || tool === "polygon") return "crosshair";
    return "default";
  };

  return (
    <div
      style={{ display: "flex", height: "100vh" ,  position: "relative" }}
    >
           <BlobLarge className="top-[-200px] left-[-400px]" />
      <div
        style={{
          width: "64px",
          backgroundColor: "black",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          paddingTop: "16px",
          gap: "8px",
        }}
      >
        <button
          onClick={() => setTool("bbox")}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: tool === "bbox" ? "#1976d2" : "#424242",
            color: tool === "bbox" ? "#fff" : "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          title="Bounding Box"
        >
          □
        </button>

        <button
          onClick={() => setTool("polygon")}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: tool === "polygon" ? "#1976d2" : "#424242",
            color: tool === "polygon" ? "#fff" : "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          title="Polygon"
        >
          ⬟
        </button>

        <button
          onClick={() => setTool("move")}
          disabled={!roi && !polygonClosed}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: tool === "move" ? "#1976d2" : "#424242",
            color: tool === "move" ? "#fff" : "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor: !roi && !polygonClosed ? "not-allowed" : "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            opacity: !roi && !polygonClosed ? 0.5 : 1,
          }}
          title="Move"
        >
          ✥
        </button>

        <button
          onClick={handleExpandRegion}
          disabled={(!roi && !polygonClosed) || tool === "pan"}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: "#424242",
            color: "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor:
              (!roi && !polygonClosed) || tool === "pan"
                ? "not-allowed"
                : "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            opacity: (!roi && !polygonClosed) || tool === "pan" ? 0.5 : 1,
          }}
          title="Expand Region"
        >
          ⊕
        </button>

        <button
          onClick={handleShrinkRegion}
          disabled={(!roi && !polygonClosed) || tool === "pan"}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: "#424242",
            color: "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor:
              (!roi && !polygonClosed) || tool === "pan"
                ? "not-allowed"
                : "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            opacity: (!roi && !polygonClosed) || tool === "pan" ? 0.5 : 1,
          }}
          title="Shrink Region"
        >
          ⊖
        </button>

        <button
          onClick={handleUndo}
          disabled={tool !== "polygon" || polygonPoints.length === 0}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: "#424242",
            color: "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor:
              tool !== "polygon" || polygonPoints.length === 0
                ? "not-allowed"
                : "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            opacity: tool !== "polygon" || polygonPoints.length === 0 ? 0.5 : 1,
          }}
          title="Undo"
        >
          ↶
        </button>

        <button
          onClick={handleErase}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: "#424242",
            color: "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          title="Eraser"
        >
          ✕
        </button>

        <div
          style={{
            width: "60%",
            height: "1px",
            backgroundColor: "#616161",
            margin: "8px 0",
          }}
        />

        <button
          onClick={() => setTool("pan")}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: tool === "pan" ? "#1976d2" : "#424242",
            color: tool === "pan" ? "#fff" : "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          title="Pan"
        >
          ✋
        </button>

        <button
          onClick={handleZoomIn}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: "#424242",
            color: "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          title="Zoom In"
        >
          +
        </button>

        <button
          onClick={handleZoomOut}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: "#424242",
            color: "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          title="Zoom Out"
        >
          −
        </button>

        <div
          style={{
            width: "60%",
            height: "1px",
            backgroundColor: "#616161",
            margin: "8px 0",
          }}
        />

        <button
          onClick={handleReset}
          style={{
            width: "40px",
            height: "40px",
            backgroundColor: "#424242",
            color: "#9e9e9e",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          title="Reset"
        >
          ↺
        </button>
      </div>

      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <div
          style={{
            padding: "16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
             bgcolor: "rgba(255, 255, 255, 0.03)",
          
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            <h3 style={{ margin: 0 }}>Region Selection Tool</h3>
            <span
              style={{
                padding: "4px 12px",
                backgroundColor: "#e3f2fd",
                color: "#1976d2",
                borderRadius: "16px",
                fontSize: "12px",
              }}
            >
              Zoom: {(zoom * 100).toFixed(0)}%
            </span>
            {tool === "polygon" && (
              <span
                style={{
                  padding: "4px 12px",
                  backgroundColor: polygonClosed ? "#e8f5e9" : "#fff3e0",
                  color: polygonClosed ? "#4caf50" : "#ff9800",
                  borderRadius: "16px",
                  fontSize: "12px",
                }}
              >
                {polygonClosed
                  ? `Polygon Closed (${polygonPoints.length} vertices)`
                  : `Drawing: ${polygonPoints.length} points`}
              </span>
            )}
            {tool === "bbox" && roi && (
              <span
                style={{
                  padding: "4px 12px",
                  backgroundColor: "#e8f5e9",
                  color: "#4caf50",
                  borderRadius: "16px",
                  fontSize: "12px",
                }}
              >
                Rectangle: {roi.width.toFixed(0)} × {roi.height.toFixed(0)}
              </span>
            )}
          </div>

          <div style={{ display: "flex", gap: "8px" }}>
            {(roi || polygonClosed) && (
              <button
                onClick={handleGenerate}
                style={{
                  padding: "8px 16px",
                  backgroundColor: "#4caf50",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                }}
              >
                ✓ Generate Mask
              </button>
            )}
         
          </div>
        </div>

        <div
          ref={containerRef}
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "16px",
            overflow: "hidden",
          }}
        >
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
            style={{
            //   backgroundColor: "white",
              boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
              maxWidth: "100%",
              maxHeight: "100%",
              cursor: getCursorStyle(),
            }}
          />
        </div>

        {tool === "polygon" && polygonPoints.length > 0 && !polygonClosed && (
          <div
            style={{
              padding: "16px",
              backgroundColor: "#2196f3",
              color: "white",
            }}
          >
            Click to add points to the polygon. Click near the first point
            (yellow circle) to close the polygon.
          </div>
        )}
      </div>

      {showResult && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
          }}
        >
          <div
            style={{
            //   backgroundColor: "white",
              borderRadius: "8px",
              padding: "24px",
              maxWidth: "600px",
              width: "90%",
              maxHeight: "80vh",
              overflow: "auto",
            }}
          >
            <h3>Generated Mask Data</h3>
            <pre
              style={{
                backgroundColor: "#f5f5f5",
                padding: "16px",
                borderRadius: "4px",
                overflow: "auto",
                fontSize: "12px",
                fontFamily: "monospace",
              }}
            >
              {resultData
                ? JSON.stringify(
                    {
                      ...resultData,
                      mask_base64:
                        resultData.mask_base64.substring(0, 50) + "...",
                    },
                    null,
                    2
                  )
                : ""}
            </pre>
            <div
              style={{
                display: "flex",
                gap: "8px",
                marginTop: "16px",
                justifyContent: "flex-end",
              }}
            >
              <button
                onClick={() => {
                  if (resultData) {
                    console.log("Full Mask Data:", resultData);
                    navigator.clipboard.writeText(JSON.stringify(resultData));
                  }
                }}
                style={{
                  padding: "8px 16px",
                  backgroundColor: "#1976d2",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Copy Full JSON
              </button>
              <button
                onClick={() => setShowResult(false)}
                style={{
                  padding: "8px 16px",
                  backgroundColor: "#757575",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageMeasurementTool;
