"""
WebSocket routes for Rocky AI
Real-time communication endpoints
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Optional
import json
import asyncio
from apps.api.app.websocket import get_connection_manager, WebSocketEventHandler, MessageType
from apps.api.app.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None),
    connection_manager = Depends(get_connection_manager)
):
    """Main WebSocket endpoint for real-time communication"""
    await connection_manager.connect(websocket, user_id)
    event_handler = WebSocketEventHandler(connection_manager)
    
    try:
        # Send welcome message
        await connection_manager.send_personal_message({
            "type": "connection_established",
            "message": "Connected to Rocky AI real-time updates",
            "user_id": user_id,
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)
        
        # Start ping/pong heartbeat
        asyncio.create_task(heartbeat_task(websocket, connection_manager))
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle the message
            await event_handler.handle_message(websocket, message)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected: {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


@router.websocket("/ws/analysis/{analysis_id}")
async def analysis_websocket_endpoint(
    websocket: WebSocket,
    analysis_id: str,
    user_id: Optional[str] = Query(None),
    connection_manager = Depends(get_connection_manager)
):
    """WebSocket endpoint for specific analysis updates"""
    await connection_manager.connect(websocket, user_id)
    connection_manager.subscribe_to_analysis(websocket, analysis_id)
    
    try:
        # Send analysis subscription confirmation
        await connection_manager.send_personal_message({
            "type": "analysis_subscribed",
            "analysis_id": analysis_id,
            "message": f"Subscribed to updates for analysis {analysis_id}",
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)
        
        # Start ping/pong heartbeat
        asyncio.create_task(heartbeat_task(websocket, connection_manager))
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle the message
            event_handler = WebSocketEventHandler(connection_manager)
            await event_handler.handle_message(websocket, message)
            
    except WebSocketDisconnect:
        connection_manager.unsubscribe_from_analysis(websocket, analysis_id)
        connection_manager.disconnect(websocket)
        logger.info(f"Analysis WebSocket disconnected: {analysis_id}")
    except Exception as e:
        logger.error(f"Analysis WebSocket error: {e}")
        connection_manager.unsubscribe_from_analysis(websocket, analysis_id)
        connection_manager.disconnect(websocket)


async def heartbeat_task(websocket: WebSocket, connection_manager):
    """Send periodic ping messages to keep connection alive"""
    try:
        while True:
            await asyncio.sleep(30)  # Send ping every 30 seconds
            await connection_manager.send_personal_message({
                "type": MessageType.PING,
                "timestamp": asyncio.get_event_loop().time()
            }, websocket)
    except Exception as e:
        logger.debug(f"Heartbeat task ended: {e}")


@router.get("/ws/stats")
async def websocket_stats(connection_manager = Depends(get_connection_manager)):
    """Get WebSocket connection statistics"""
    return connection_manager.get_connection_stats()
