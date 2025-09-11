import React, { useEffect, useState } from "react";

const API_BASE = "http://localhost:5000"; // Flask server URL
const STATUS_URL = `${API_BASE}/api/status`;
const PUBLISH_URL = `${API_BASE}/api/publish_detection`;

export default function MqttDashboard() {
  const [messages, setMessages] = useState([]);
  const [connected, setConnected] = useState(false);
  const [viewMode, setViewMode] = useState("csv"); // csv | json
  const [stats, setStats] = useState({
    total: 0,
    detections: 0,
    tests: 0,
    lastUpdate: "Never",
  });

  // Fetch messages
  const fetchMessages = async () => {
    try {
      const res = await fetch(STATUS_URL);
      if (!res.ok) throw new Error("Failed to fetch");
      const data = await res.json();
      const msgs = data.messages || [];
      setMessages(msgs);
      setConnected(true);
      updateStats(msgs);
    } catch (err) {
      console.error(err);
      setConnected(false);
    }
  };

  // Update statistics
  const updateStats = (msgs) => {
    const total = msgs.length;
    const detections = msgs.filter((m) => m.topic === "detection/results").length;
    const tests = msgs.filter((m) => m.topic === "test/topic").length;
    const lastUpdate =
      msgs.length > 0
        ? new Date(
            Math.max(...msgs.map((m) => new Date(m.timestamp)))
          ).toLocaleString()
        : "Never";

    setStats({ total, detections, tests, lastUpdate });
  };

  // Toggle view mode
  const toggleView = () => {
    setViewMode(viewMode === "csv" ? "json" : "csv");
  };

  // Render payloads
  const renderCSV = (data) => {
    if (!data || typeof data !== "object") return <i>Invalid payload</i>;

    // Raw Dough
    if ("number_of_defected" in data && "number_of_good" in data) {
      return (
        <table className="w-full text-sm border-collapse border">
          <thead>
            <tr>
              <th>Tray</th>
              <th>Good</th>
              <th>Defected</th>
              <th>Missing</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{data.tray_number}</td>
              <td>{data.number_of_good}</td>
              <td>{data.number_of_defected}</td>
              <td>{data.number_of_missing}</td>
              <td>{data.total_detected}</td>
            </tr>
          </tbody>
        </table>
      );
    }

    // Baked Baguette
    if ("good" in data && "defected" in data) {
      return (
        <table className="w-full text-sm border-collapse border">
          <thead>
            <tr>
              <th>Tray</th>
              <th>Good</th>
              <th>Acceptable</th>
              <th>Defected</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{data.tray_number}</td>
              <td>{data.good}</td>
              <td>{data.acceptable}</td>
              <td>{data.defected}</td>
              <td>{data.total_detected}</td>
            </tr>
          </tbody>
        </table>
      );
    }

    // Donuts
    if ("doughnuts" in data) {
      return (
        <table className="w-full text-sm border-collapse border">
          <thead>
            <tr>
              <th>Row</th>
              <th>Doughnut #</th>
              <th>Status</th>
              <th>Measurement Diff</th>
            </tr>
          </thead>
          <tbody>
            {data.doughnuts.map((d, i) => (
              <tr key={i}>
                <td>{data.row_number}</td>
                <td>{d.doughnut_number}</td>
                <td>{d.status}</td>
                <td>{d.measurement_difference}</td>
              </tr>
            ))}
          </tbody>
        </table>
      );
    }

    // Fallback
    return <pre>{JSON.stringify(data, null, 2)}</pre>;
  };

  // Auto-refresh
  useEffect(() => {
    fetchMessages();
    const interval = setInterval(fetchMessages, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className=" bg-slate-900 text-slate-200 rounded-xl shadow-lg">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-slate-600 pb-4 mb-4">
        <h1 className="text-2xl font-bold text-blue-400">MQTT Results:</h1>
        <div
          className={`flex items-center gap-2 px-4 py-2 rounded-full ${
            connected ? "bg-green-600/20" : "bg-red-600/20"
          }`}
        >
          <span
            className={`w-3 h-3 rounded-full ${
              connected ? "bg-green-500" : "bg-red-500"
            }`}
          ></span>
          {connected ? "Connected" : "Disconnected"}
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-3 mb-4">
        <button
          className="px-3 py-2 rounded bg-blue-600 hover:bg-blue-500"
          onClick={fetchMessages}
        >
          🔄 Refresh
        </button>
        <button
          className="px-3 py-2 rounded bg-indigo-600 hover:bg-indigo-500"
          onClick={toggleView}
        >
          {viewMode === "csv" ? "📋 JSON View" : "📊 CSV View"}
        </button>
      </div>

      {/* Messages */}
      <div className="space-y-3 max-h-[400px] overflow-y-auto">
        {messages.length === 0 ? (
          <div className="italic text-slate-400">No messages yet</div>
        ) : (
          [...messages]
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .map((msg, idx) => {
              let payload;
              try {
                const parsed = JSON.parse(msg.payload);
                payload =
                  viewMode === "json" ? (
                    <pre>{JSON.stringify(parsed, null, 2)}</pre>
                  ) : (
                    renderCSV(parsed)
                  );
              } catch {
                payload = msg.payload;
              }

              return (
                <div
                  key={idx}
                  className="p-3 bg-slate-800 border border-slate-700 rounded-md"
                >
                  <div className="text-blue-400 font-semibold">
                    📡 {msg.topic}
                  </div>
                  <div className="mt-2">{payload}</div>
                  <div className="text-xs text-slate-400 mt-2 flex justify-between">
                    <span>🕐 {new Date(msg.timestamp).toLocaleString()}</span>
                    <span>📊 QoS: {msg.qos || 0}</span>
                  </div>
                </div>
              );
            })
        )}
      </div>

      {/* Stats */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-3 bg-slate-800 rounded text-center">
          <div className="text-2xl font-bold text-blue-400">
            {stats.total}
          </div>
          <div className="text-xs text-slate-400">Total Messages</div>
        </div>
        <div className="p-3 bg-slate-800 rounded text-center">
          <div className="text-2xl font-bold text-blue-400">
            {stats.detections}
          </div>
          <div className="text-xs text-slate-400">Detections</div>
        </div>
        <div className="p-3 bg-slate-800 rounded text-center">
          <div className="text-2xl font-bold text-blue-400">{stats.tests}</div>
          <div className="text-xs text-slate-400">Test Messages</div>
        </div>
        <div className="p-3 bg-slate-800 rounded text-center">
          <div className="text-sm font-semibold">{stats.lastUpdate}</div>
          <div className="text-xs text-slate-400">Last Update</div>
        </div>
      </div>
    </div>
  );
}
