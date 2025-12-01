# Roadmap

## Future Enhancements

### Conversation Persistence Across Timeouts

**Current Behavior:**
When a 20-second silence timeout occurs, the device disconnects from the proxy and the OpenAI Realtime API connection is terminated. When the user reconnects, a fresh conversation starts with no memory of the previous context.

**Desired Behavior:**
Maintain conversation history across timeout events so users can resume their conversation seamlessly after periods of silence.

**Implementation Requirements:**

1. **Session Management in Proxy:**
   - Keep OpenAI Realtime API connections alive when device disconnects due to timeout
   - Map device `session_id` to persistent OpenAI sessions
   - Implement session timeout policy (e.g., close OpenAI connection after 5 minutes of inactivity)

2. **Device Changes:**
   - Send `session_id` on reconnection to resume existing session
   - Handle session expiry gracefully (start fresh if session no longer exists)

3. **Proxy API Changes:**
   - Add session resume capability to WebSocket endpoint
   - Distinguish between "new session" and "resume session" requests
   - Track session state (active, paused, expired)

4. **Edge Cases to Handle:**
   - Session expired while device disconnected → start fresh
   - Multiple devices with same session_id → handle gracefully
   - Memory management for long-running sessions
   - Conversation history limits (token count considerations)

**Benefits:**
- More natural conversation flow
- Users can take breaks without losing context
- Better user experience for voice assistants

**Considerations:**
- OpenAI API costs (keeping connections alive)
- Memory usage on proxy server
- When to truly end a conversation vs. pause it
- User expectations around conversation persistence

---

**Status:** Planned for future implementation
**Priority:** Medium
**Estimated Effort:** 2-3 days
