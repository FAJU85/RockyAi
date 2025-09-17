"""
WebSocket service for real-time updates in Rocky AI
Handles real-time communication between frontend and backend
"""
import json
import asyncio
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from apps.api.app.logging_config import get_logger
from apps.api.app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class ConnectionManager:
    """Manages WebSocket connections and real-time updates"""
    
    def __init__(self):
        # Store active connections by user ID
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Store connections by analysis ID for targeted updates
        self.analysis_connections: Dict[str, Set[WebSocket]] = {}
        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        # Store connection metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": asyncio.get_event_loop().time(),
            "last_ping": asyncio.get_event_loop().time()
        }
        
        # Add to user connections if user_id provided
        if user_id:
            if user_id not in self.active_connections:
                self.active_connections[user_id] = set()
            self.active_connections[user_id].add(websocket)
        
        logger.info(f"WebSocket connected for user: {user_id}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        metadata = self.connection_metadata.get(websocket, {})
        user_id = metadata.get("user_id")
        
        # Remove from user connections
        if user_id and user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Remove from analysis connections
        for analysis_id, connections in self.analysis_connections.items():
            connections.discard(websocket)
            if not connections:
                del self.analysis_connections[analysis_id]
        
        # Remove metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        
        logger.info(f"WebSocket disconnected for user: {user_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def send_to_user(self, message: Dict[str, Any], user_id: str):
        """Send a message to all connections for a specific user"""
        if user_id in self.active_connections:
            disconnected = set()
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send message to user {user_id}: {e}")
                    disconnected.add(websocket)
            
            # Clean up disconnected connections
            for websocket in disconnected:
                self.disconnect(websocket)
    
    async def send_to_analysis(self, message: Dict[str, Any], analysis_id: str):
        """Send a message to all connections following a specific analysis"""
        if analysis_id in self.analysis_connections:
            disconnected = set()
            for websocket in self.analysis_connections[analysis_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send message to analysis {analysis_id}: {e}")
                    disconnected.add(websocket)
            
            # Clean up disconnected connections
            for websocket in disconnected:
                self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        disconnected = set()
        for user_id, connections in self.active_connections.items():
            for websocket in connections:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to broadcast message: {e}")
                    disconnected.add(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected:
            self.disconnect(websocket)
    
    def subscribe_to_analysis(self, websocket: WebSocket, analysis_id: str):
        """Subscribe a connection to updates for a specific analysis"""
        if analysis_id not in self.analysis_connections:
            self.analysis_connections[analysis_id] = set()
        self.analysis_connections[analysis_id].add(websocket)
        logger.debug(f"Subscribed to analysis {analysis_id}")
    
    def unsubscribe_from_analysis(self, websocket: WebSocket, analysis_id: str):
        """Unsubscribe a connection from updates for a specific analysis"""
        if analysis_id in self.analysis_connections:
            self.analysis_connections[analysis_id].discard(websocket)
            if not self.analysis_connections[analysis_id]:
                del self.analysis_connections[analysis_id]
        logger.debug(f"Unsubscribed from analysis {analysis_id}")
    
    async def send_analysis_update(self, analysis_id: str, update_type: str, data: Dict[str, Any]):
        """Send an analysis update to subscribed connections"""
        message = {
            "type": "analysis_update",
            "analysis_id": analysis_id,
            "update_type": update_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.send_to_analysis(message, analysis_id)
    
    async def send_system_notification(self, notification_type: str, message: str, user_id: Optional[str] = None):
        """Send a system notification"""
        notification = {
            "type": "system_notification",
            "notification_type": notification_type,
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        if user_id:
            await self.send_to_user(notification, user_id)
        else:
            await self.broadcast(notification)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total_connections = sum(len(connections) for connections in self.active_connections.values())
        total_analysis_subscriptions = sum(len(connections) for connections in self.analysis_connections.values())
        
        return {
            "total_connections": total_connections,
            "active_users": len(self.active_connections),
            "analysis_subscriptions": total_analysis_subscriptions,
            "unique_analyses": len(self.analysis_connections)
        }


# Global connection manager
manager = ConnectionManager()


async def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager"""
    return manager


# WebSocket message types
class MessageType:
    ANALYSIS_START = "analysis_start"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_ERROR = "analysis_error"
    SYSTEM_NOTIFICATION = "system_notification"
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE_ANALYSIS = "subscribe_analysis"
    UNSUBSCRIBE_ANALYSIS = "unsubscribe_analysis"


# WebSocket event handlers
class WebSocketEventHandler:
    """Handles WebSocket events and message routing"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
    
    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        message_type = message.get("type")
        
        try:
            if message_type == MessageType.PING:
                await self.handle_ping(websocket, message)
            elif message_type == MessageType.SUBSCRIBE_ANALYSIS:
                await self.handle_subscribe_analysis(websocket, message)
            elif message_type == MessageType.UNSUBSCRIBE_ANALYSIS:
                await self.handle_unsubscribe_analysis(websocket, message)
            else:
                logger.warning(f"Unknown message type: {message_type}")
        
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.manager.send_personal_message({
                "type": "error",
                "message": "Failed to process message",
                "error": str(e)
            }, websocket)
    
    async def handle_ping(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle ping messages"""
        await self.manager.send_personal_message({
            "type": MessageType.PONG,
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)
    
    async def handle_subscribe_analysis(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle analysis subscription requests"""
        analysis_id = message.get("analysis_id")
        if analysis_id:
            self.manager.subscribe_to_analysis(websocket, analysis_id)
            await self.manager.send_personal_message({
                "type": "subscription_confirmed",
                "analysis_id": analysis_id,
                "status": "subscribed"
            }, websocket)
    
    async def handle_unsubscribe_analysis(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle analysis unsubscription requests"""
        analysis_id = message.get("analysis_id")
        if analysis_id:
            self.manager.unsubscribe_from_analysis(websocket, analysis_id)
            await self.manager.send_personal_message({
                "type": "unsubscription_confirmed",
                "analysis_id": analysis_id,
                "status": "unsubscribed"
            }, websocket)
