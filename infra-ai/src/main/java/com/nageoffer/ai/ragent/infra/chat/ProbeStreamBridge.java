/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nageoffer.ai.ragent.infra.chat;

import com.nageoffer.ai.ragent.framework.errorcode.BaseErrorCode;
import com.nageoffer.ai.ragent.framework.exception.RemoteException;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 流式首包探测桥接器
 */
final class ProbeStreamBridge implements StreamCallback {

    private final StreamCallback downstream;
    private final Object lock = new Object();
    private final List<BufferedEvent> bufferedEvents = new ArrayList<>();
    private volatile boolean committed;

    private final CountDownLatch latch = new CountDownLatch(1);
    private final AtomicBoolean hasContent = new AtomicBoolean(false);
    private final AtomicBoolean eventFired = new AtomicBoolean(false);
    private final AtomicReference<Throwable> error = new AtomicReference<>();

    ProbeStreamBridge(StreamCallback downstream) {
        this.downstream = downstream;
        this.committed = false;
    }

    // ==================== StreamCallback 实现 ====================

    @Override
    public void onContent(String content) {
        markContent();
        bufferOrDispatch(BufferedEvent.content(content));
    }

    @Override
    public void onThinking(String content) {
        markContent();
        bufferOrDispatch(BufferedEvent.thinking(content));
    }

    @Override
    public void onComplete() {
        markComplete();
        bufferOrDispatch(BufferedEvent.complete());
    }

    @Override
    public void onError(Throwable t) {
        markError(t);
        bufferOrDispatch(BufferedEvent.error(t));
    }

    // ==================== 探测等待 ====================

    /**
     * 阻塞等待首包探测结果，SUCCESS 时自动提交缓冲
     */
    ProbeResult awaitFirstPacket(long timeout, TimeUnit unit) throws InterruptedException {
        boolean completed = latch.await(timeout, unit);

        if (error.get() != null) {
            return ProbeResult.error(error.get());
        }
        if (!completed) {
            return ProbeResult.timeout();
        }
        if (!hasContent.get()) {
            return ProbeResult.noContent();
        }

        commit();
        return ProbeResult.success();
    }

    // ==================== 内部方法 ====================

    private void markContent() {
        hasContent.set(true);
        fireEventOnce();
    }

    private void markComplete() {
        fireEventOnce();
    }

    private void markError(Throwable t) {
        error.set(t);
        fireEventOnce();
    }

    private void fireEventOnce() {
        if (eventFired.compareAndSet(false, true)) {
            latch.countDown();
        }
    }

    private void commit() {
        synchronized (lock) {
            if (committed) {
                return;
            }
            committed = true;
            for (BufferedEvent event : bufferedEvents) {
                dispatch(event);
            }
        }
    }

    private void bufferOrDispatch(BufferedEvent event) {
        boolean dispatchNow;
        synchronized (lock) {
            dispatchNow = committed;
            if (!dispatchNow) {
                bufferedEvents.add(event);
            }
        }
        if (dispatchNow) {
            dispatch(event);
        }
    }

    private void dispatch(BufferedEvent event) {
        switch (event.type()) {
            case CONTENT -> downstream.onContent(event.content());
            case THINKING -> downstream.onThinking(event.content());
            case COMPLETE -> downstream.onComplete();
            case ERROR -> downstream.onError(event.error() != null
                    ? event.error()
                    : new RemoteException("流式请求失败", BaseErrorCode.REMOTE_ERROR));
        }
    }

    // ==================== 内部数据结构 ====================

    private record BufferedEvent(EventType type, String content, Throwable error) {

        private static BufferedEvent content(String content) {
            return new BufferedEvent(EventType.CONTENT, content, null);
        }

        private static BufferedEvent thinking(String content) {
            return new BufferedEvent(EventType.THINKING, content, null);
        }

        private static BufferedEvent complete() {
            return new BufferedEvent(EventType.COMPLETE, null, null);
        }

        private static BufferedEvent error(Throwable error) {
            return new BufferedEvent(EventType.ERROR, null, error);
        }
    }

    private enum EventType {
        CONTENT,
        THINKING,
        COMPLETE,
        ERROR
    }

    /**
     * 探测结果
     */
    @Getter
    static class ProbeResult {

        enum Type {SUCCESS, ERROR, TIMEOUT, NO_CONTENT}

        private final Type type;
        private final Throwable error;

        private ProbeResult(Type type, Throwable error) {
            this.type = type;
            this.error = error;
        }

        static ProbeResult success() {
            return new ProbeResult(Type.SUCCESS, null);
        }

        static ProbeResult error(Throwable t) {
            return new ProbeResult(Type.ERROR, t);
        }

        static ProbeResult timeout() {
            return new ProbeResult(Type.TIMEOUT, null);
        }

        static ProbeResult noContent() {
            return new ProbeResult(Type.NO_CONTENT, null);
        }

        boolean isSuccess() {
            return type == Type.SUCCESS;
        }
    }
}
