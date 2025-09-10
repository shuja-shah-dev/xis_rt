import React, { useState, useRef, useEffect } from "react";
import {
  Box,
  Paper,
  IconButton,
  Button,
  Typography,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Tooltip,
  Divider,
  Chip,
} from "@mui/material";
import {
  CropFree,
  ClearAll,
  PanTool,
  ZoomIn,
  ZoomOut,
  RestartAlt,
  PushPin,
  CloudUpload,
  ArrowForward,
  OpenWith,
} from "@mui/icons-material";
import RoundedButton from "./RoundedButton";
import { BlobLarge } from "./Blob";

const ImageMeasurementTool = ({ setActiveScreen, setShowMeasurement }) => {
  const [stage, setStage] = useState("roi");
  const [tool, setTool] = useState("bbox");
  const [imageUrl, setImageUrl] = useState(
    "https://images.unsplash.com/photo-1517411029-9ef8ca894182?w=800"
  );
  const [imageLoaded, setImageLoaded] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [isMoving, setIsMoving] = useState(false);
  const [startPoint, setStartPoint] = useState(null);
  const [endPoint, setEndPoint] = useState(null);
  const [roi, setRoi] = useState(null);
  const [keypoints, setKeypoints] = useState([]);
  const [lengthMm, setLengthMm] = useState("");
  const [pixelDistance, setPixelDistance] = useState(null);
  const [showInput, setShowInput] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [lastPanPoint, setLastPanPoint] = useState(null);
  const [moveStartPoint, setMoveStartPoint] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });
  const [initialScale, setInitialScale] = useState(1);

  const canvasRef = useRef(null);
  const imageRef = useRef(new Image());
  const containerRef = useRef(null);

  const handleContinue = () => {
    setShowMeasurement(false);
    setActiveScreen("Inference");
  };

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
  }, [imageLoaded, roi, keypoints, startPoint, endPoint, zoom, pan, stage]);

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

    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    const drawX = (canvasSize.width / zoom - canvasSize.width) / 2;
    const drawY = (canvasSize.height / zoom - canvasSize.height) / 2;

    ctx.drawImage(
      imageRef.current,
      0,
      0,
      imageRef.current.width,
      imageRef.current.height,
      drawX,
      drawY,
      canvasSize.width / zoom,
      canvasSize.height / zoom
    );

    if (stage === "roi") {
      if (roi) {
        ctx.strokeStyle = "#4caf50";
        ctx.lineWidth = 2 / zoom;
        ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
        ctx.fillStyle = "rgba(76, 175, 80, 0.1)";
        ctx.fillRect(roi.x, roi.y, roi.width, roi.height);
      }

      if (isDrawing && startPoint && endPoint) {
        ctx.strokeStyle = "#4caf50";
        ctx.lineWidth = 2 / zoom;
        ctx.setLineDash([5 / zoom, 5 / zoom]);
        const width = endPoint.x - startPoint.x;
        const height = endPoint.y - startPoint.y;
        ctx.strokeRect(startPoint.x, startPoint.y, width, height);
        ctx.setLineDash([]);
      }
    } else if (stage === "keypoints") {
      if (roi) {
        ctx.strokeStyle = "#4caf50";
        ctx.lineWidth = 1 / zoom;
        ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
        ctx.fillStyle = "rgba(76, 175, 80, 0.05)";
        ctx.fillRect(roi.x, roi.y, roi.width, roi.height);
      }

      keypoints.forEach((point, index) => {
        ctx.fillStyle = "#f44336";
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5 / zoom, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2 / zoom;
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.font = `${12 / zoom}px Arial`;
        ctx.fillText(`${index + 1}`, point.x - 4 / zoom, point.y + 4 / zoom);
      });

      if (keypoints.length === 2) {
        ctx.strokeStyle = "#ffc107";
        ctx.lineWidth = 2 / zoom;
        ctx.beginPath();
        ctx.moveTo(keypoints[0].x, keypoints[0].y);
        ctx.lineTo(keypoints[1].x, keypoints[1].y);
        ctx.stroke();

        const midX = (keypoints[0].x + keypoints[1].x) / 2;
        const midY = (keypoints[0].y + keypoints[1].y) / 2;
        const dist = Math.sqrt(
          Math.pow(keypoints[1].x - keypoints[0].x, 2) +
            Math.pow(keypoints[1].y - keypoints[0].y, 2)
        );

        ctx.fillStyle = "#000000";
        ctx.font = `${12 / zoom}px Arial`;
        ctx.fillText(`${dist.toFixed(1)}px`, midX + 5 / zoom, midY - 5 / zoom);
      }
    }

    ctx.restore();
  };

  const screenToCanvas = (screenX, screenY) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (screenX - rect.left - pan.x) / zoom;
    const y = (screenY - rect.top - pan.y) / zoom;
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

  const handleMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    const pos = screenToCanvas(e.clientX, e.clientY);

    if (tool === "pan") {
      setIsPanning(true);
      setLastPanPoint({ x: e.clientX, y: e.clientY });
    } else if (tool === "move" && stage === "roi" && roi) {
      if (isPointInRoi(pos, roi)) {
        setIsMoving(true);
        setMoveStartPoint(pos);
      }
    } else if (tool === "bbox" && stage === "roi") {
      setStartPoint(pos);
      setEndPoint(pos);
      setIsDrawing(true);
    } else if (
      tool === "keypoint" &&
      stage === "keypoints" &&
      keypoints.length < 2
    ) {
      const newKeypoints = [...keypoints, pos];
      setKeypoints(newKeypoints);

      if (newKeypoints.length === 2) {
        const dist = Math.sqrt(
          Math.pow(newKeypoints[1].x - newKeypoints[0].x, 2) +
            Math.pow(newKeypoints[1].y - newKeypoints[0].y, 2)
        );
        setPixelDistance(dist);
        setShowInput(true);
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
    } else if (isMoving && moveStartPoint && roi) {
      const dx = pos.x - moveStartPoint.x;
      const dy = pos.y - moveStartPoint.y;
      setRoi({
        ...roi,
        x: roi.x + dx,
        y: roi.y + dy,
      });
      setMoveStartPoint(pos);
    } else if (isDrawing && tool === "bbox" && stage === "roi") {
      setEndPoint(pos);
    }
  };

  const handleMouseUp = () => {
    if (isPanning) {
      setIsPanning(false);
      setLastPanPoint(null);
    } else if (isMoving) {
      setIsMoving(false);
      setMoveStartPoint(null);
    } else if (
      isDrawing &&
      tool === "bbox" &&
      stage === "roi" &&
      startPoint &&
      endPoint
    ) {
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

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev * 1.2, 5));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev / 1.2, 0.5));
  };

  const handleNext = () => {
    if (stage === "roi" && roi) {
      setStage("keypoints");
      setTool("keypoint");
      setKeypoints([]);
      setPixelDistance(null);
      setLengthMm("");
    }
  };

  const handleReset = () => {
    setStage("roi");
    setTool("bbox");
    setRoi(null);
    setKeypoints([]);
    setPixelDistance(null);
    setLengthMm("");
    setShowInput(false);
    setStartPoint(null);
    setEndPoint(null);
    setIsMoving(false);
    setMoveStartPoint(null);
    fitImageToCanvas();
  };

  const handleErase = () => {
    if (stage === "roi") {
      setRoi(null);
    } else if (stage === "keypoints") {
      setKeypoints([]);
      setPixelDistance(null);
    }
  };

  const handleCalculate = () => {
    if (lengthMm && pixelDistance && roi) {
      const mmPerPixel = parseFloat(lengthMm) / pixelDistance;

      const originalRoi = canvasToOriginalImage(roi.x, roi.y);
      const originalRoiWidth = roi.width / initialScale;
      const originalRoiHeight = roi.height / initialScale;

      const roiCenterX = originalRoi.x + originalRoiWidth / 2;
      const roiCenterY = originalRoi.y + originalRoiHeight / 2;

      const originalPixelDistance = pixelDistance / initialScale;

      const resultData = {
        ROI: {
          x: roiCenterX.toFixed(2),
          y: roiCenterY.toFixed(2),
          height: originalRoiHeight.toFixed(2),
          width: originalRoiWidth.toFixed(2),
        },
        reference: {
          pixel_length: originalPixelDistance.toFixed(2),
          actual_length: lengthMm,
        },
      };

      console.log("Measurement Result:", JSON.stringify(resultData, null, 2));

      alert(
        `Calibration Complete!\nPixel Distance: ${originalPixelDistance.toFixed(
          2
        )} px\nActual Length: ${lengthMm} mm\nScale: ${mmPerPixel.toFixed(
          4
        )} mm/pixel`
      );

      setShowInput(false);
    }
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
    if (tool === "bbox" || tool === "keypoint") return "crosshair";
    return "default";
  };

  return (
    <>
      <Box sx={{ display: "flex", height: "100vh", position: "relative" }}>
        <BlobLarge className="top-[-200px] left-[-400px]" />
        {/* Left Sidebar */}
        <Paper
          elevation={3}
          sx={{
            width: 64,
            bgcolor: "black",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            py: 2,
            gap: 1,
            borderRadius: 0,
          }}
        >
          <Tooltip title="Bounding Box" placement="right">
            <span>
              <IconButton
                onClick={() => setTool("bbox")}
                disabled={stage !== "roi"}
                sx={{
                  color:
                    tool === "bbox" && stage === "roi" ? "#ffffff" : "#a0aec0",
                  bgcolor:
                    tool === "bbox" && stage === "roi"
                      ? "#4a5568"
                      : "transparent",
                  "&:hover": { bgcolor: "#4a5568" },
                  "&:disabled": { color: "#718096" },
                }}
              >
                <CropFree />
              </IconButton>
            </span>
          </Tooltip>

          <Tooltip title="Move ROI" placement="right">
            <span>
              <IconButton
                onClick={() => setTool("move")}
                disabled={stage !== "roi" || !roi}
                sx={{
                  color:
                    tool === "move" && stage === "roi" ? "#ffffff" : "#a0aec0",
                  bgcolor:
                    tool === "move" && stage === "roi"
                      ? "#4a5568"
                      : "transparent",
                  "&:hover": { bgcolor: "#4a5568" },
                  "&:disabled": { color: "#718096" },
                }}
              >
                <OpenWith />
              </IconButton>
            </span>
          </Tooltip>

          <Tooltip title="Keypoint" placement="right">
            <span>
              <IconButton
                onClick={() => setTool("keypoint")}
                disabled={stage !== "keypoints"}
                sx={{
                  color:
                    tool === "keypoint" && stage === "keypoints"
                      ? "#ffffff"
                      : "#a0aec0",
                  bgcolor:
                    tool === "keypoint" && stage === "keypoints"
                      ? "#4a5568"
                      : "transparent",
                  "&:hover": { bgcolor: "#4a5568" },
                  "&:disabled": { color: "#718096" },
                }}
              >
                <PushPin />
              </IconButton>
            </span>
          </Tooltip>

          <Tooltip title="Eraser" placement="right">
            <IconButton
              onClick={handleErase}
              sx={{
                color: "#a0aec0",
                "&:hover": { bgcolor: "#4a5568" },
              }}
            >
              <ClearAll />
            </IconButton>
          </Tooltip>

          <Divider sx={{ width: "60%", bgcolor: "#4a5568", my: 1 }} />

          <Tooltip title="Pan" placement="right">
            <IconButton
              onClick={() => setTool("pan")}
              sx={{
                color: tool === "pan" ? "#ffffff" : "#a0aec0",
                bgcolor: tool === "pan" ? "#4a5568" : "transparent",
                "&:hover": { bgcolor: "#4a5568" },
              }}
            >
              <PanTool />
            </IconButton>
          </Tooltip>

          <Tooltip title="Zoom In" placement="right">
            <IconButton
              onClick={handleZoomIn}
              sx={{
                color: "#a0aec0",
                "&:hover": { bgcolor: "#4a5568" },
              }}
            >
              <ZoomIn />
            </IconButton>
          </Tooltip>

          <Tooltip title="Zoom Out" placement="right">
            <IconButton
              onClick={handleZoomOut}
              sx={{
                color: "#a0aec0",
                "&:hover": { bgcolor: "#4a5568" },
              }}
            >
              <ZoomOut />
            </IconButton>
          </Tooltip>

          <Divider sx={{ width: "60%", bgcolor: "#4a5568", my: 1 }} />

          <Tooltip title="Reset" placement="right">
            <IconButton
              onClick={handleReset}
              sx={{
                color: "#a0aec0",
                "&:hover": { bgcolor: "#e53e3e", color: "#ffffff" },
              }}
            >
              <RestartAlt />
            </IconButton>
          </Tooltip>
        </Paper>

        {/* Main Content Area */}
        <Box sx={{ flex: 1, display: "flex", flexDirection: "column" }}>
          {/* Header Bar */}
          <Paper
            elevation={2}
            sx={{
              p: 2,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              borderRadius: 0,
              bgcolor: "rgba(255, 255, 255, 0.03)",
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <Typography variant="h6" fontWeight="normal" color="white">
                {stage === "roi"
                  ? "Step 1: Draw ROI"
                  : "Step 2: Mark Keypoints"}
              </Typography>
              <Chip
                label={`Zoom: ${(zoom * 100).toFixed(0)}%`}
                size="small"
                sx={{ bgcolor: "#edf2f7", color: "#2d3748" }}
              />
              {stage === "keypoints" && (
                <Chip
                  label={`Points: ${keypoints.length}/2${
                    pixelDistance
                      ? ` | Distance: ${pixelDistance.toFixed(2)}px`
                      : ""
                  }`}
                  size="small"
                  sx={{ bgcolor: "#edf2f7", color: "#2d3748" }}
                />
              )}
            </Box>

            <Box sx={{ display: "flex", gap: 4, alignItems: "center" }}>
              {stage === "roi" && roi && (
                <RoundedButton
                  label="Next"
                  onClick={handleNext}
                  active={true}
                />
              )}
              <Tooltip title="Use this image" placement="bottom">
                <Box
                  onClick={() => {
                    setImageUrl("/frame_1.png");
                    handleReset();
                  }}
                  sx={{
                    width: 60,
                    height: 60,
                    borderRadius: 1,
                    overflow: "hidden",
                    cursor: "pointer",
                    "&:hover": {
                      borderColor: "white",
                      transform: "scale(1.05)",
                    },
                    transition: "all 0.2s ease",
                  }}
                >
                  <img
                    src="/frame_1.png"
                    alt="Sample"
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                    }}
                  />
                </Box>
              </Tooltip>
            </Box>
          </Paper>

          {/* Canvas Area */}
          <Box
            ref={containerRef}
            sx={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              p: 2,
              overflow: "hidden",
            }}
          >
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{
                // backgroundColor: "white",
                boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
                maxWidth: "100%",
                maxHeight: "100%",
                cursor: getCursorStyle(),
              }}
            />
          </Box>
        </Box>

        {/* Input Dialog */}
        <Dialog
          open={showInput}
          onClose={() => setShowInput(false)}
          PaperProps={{
            sx: {
              borderRadius: 2,
              bgcolor: "black",
              color: "white",
              border: "1px solid #0E2332",
            },
          }}
        >
          <DialogTitle sx={{ fontWeightL: "normal" }}>
            Enter Actual Length
          </DialogTitle>
          <DialogContent sx={{ bgcolor: "black", pt: 2, mt: 2 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <TextField
                autoFocus
                type="number"
                value={lengthMm}
                onChange={(e) => setLengthMm(e.target.value)}
                placeholder="Length"
                variant="outlined"
                size="small"
                sx={{ bgcolor: "white" }}
              />
              <Typography>mm</Typography>
            </Box>
          </DialogContent>
          <DialogActions sx={{ bgcolor: "black", p: 2 }}>
            <Button
              onClick={() => setShowInput(false)}
              sx={{ color: "#718096" }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleCalculate}
              variant="contained"
              disabled={!lengthMm}
              sx={{
                bgcolor: "#1272E5",
                "&:hover": { bgcolor: "#2c5282" },
                "&:disabled": { bgcolor: "#cbd5e0" },
              }}
            >
              Calculate
            </Button>
          </DialogActions>
        </Dialog>
      </Box>

      {/* Next Button */}
      <Box sx={{ p: 2, display: "flex", justifyContent: "flex-end" }}>
        <RoundedButton label="INFER" onClick={handleContinue} active={true} />
      </Box>
    </>
  );
};

export default ImageMeasurementTool;
