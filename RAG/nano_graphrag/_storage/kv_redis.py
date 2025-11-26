from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import json as std_json

try:
    import orjson # a faster json library needed for the long json files
except ImportError:  # pragma: no cover - optional dependency
    orjson = None

from redis import asyncio as redis_async

from .._utils import logger
from ..base import BaseKVStorage


def _encode_payload(value: Any) -> bytes:
    """Serialize values to UTF-8 JSON, preferring orjson when available."""
    if orjson is not None:
        return orjson.dumps(value)
    return std_json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _decode_payload(raw: bytes | None) -> Any | None:
    if raw is None:
        return None
    if orjson is not None:
        return orjson.loads(raw)
    return std_json.loads(raw.decode("utf-8"))


@dataclass
class KvRedis(BaseKVStorage[Any]):
    """Redis-backed implementation of BaseKVStorage."""

    _redis: redis_async.Redis = field(init=False, repr=False)
    _redis_prefix: str = field(init=False)
    _namespace_prefix: str = field(init=False)

    def __post_init__(self) -> None:
        redis_url = self.global_config.get("redis_url") or "redis://localhost:6379/0"
        redis_prefix = self.global_config.get("redis_prefix") or "ngr"
        self._redis = redis_async.from_url(redis_url, decode_responses=False)
        self._redis_prefix = redis_prefix
        self._namespace_prefix = f"{redis_prefix}:{self.namespace}:"
        logger.info(
            "Initialised KvRedis for namespace '%s' at %s with prefix '%s'",
            self.namespace,
            redis_url,
            redis_prefix,
        )

    def _make_key(self, item_id: str) -> str:
        return f"{self._namespace_prefix}{item_id}"

    def _extract_id(self, key: bytes | str) -> str:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if key.startswith(self._namespace_prefix):
            return key[len(self._namespace_prefix) :]
        return key.rsplit(":", 1)[-1]

    async def all_keys(self) -> list[str]:
        cursor = 0
        pattern = f"{self._namespace_prefix}*"
        keys: list[str] = []
        while True:
            cursor, batch = await self._redis.scan(cursor=cursor, match=pattern, count=1024)
            if batch:
                keys.extend(self._extract_id(k) for k in batch)
            if cursor == 0:
                break
        return keys

    async def index_done_callback(self) -> None:
        logger.debug("KvRedis index_done_callback noop for namespace '%s'", self.namespace)

    async def get_by_id(self, id: str) -> Any | None:
        raw = await self._redis.get(self._make_key(id))
        return _decode_payload(raw)

    async def get_by_ids(
        self,
        ids: list[str],
        fields: Iterable[str] | None = None,
    ) -> list[Any | None]:
        if not ids:
            return []
        keys = [self._make_key(item_id) for item_id in ids]
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.mget(keys)
            (raw_values,) = await pipe.execute()
        results: list[Any | None] = []
        for raw in raw_values:
            value = _decode_payload(raw)
            if value is None or fields is None:
                results.append(value)
                continue
            if isinstance(value, dict):
                results.append({k: value[k] for k in fields if k in value})
            else:
                results.append(value)
        return results

    async def filter_keys(self, data: list[str]) -> set[str]:
        if not data:
            return set()
        async with self._redis.pipeline(transaction=False) as pipe:
            for item_id in data:
                pipe.exists(self._make_key(item_id))
            existence = await pipe.execute()
        missing = {item_id for item_id, exists in zip(data, existence) if not exists}
        return missing
    
    async def upsert(self, data: dict[str, Any]) -> None:
        if not data:
            return
        async with self._redis.pipeline(transaction=False) as pipe:
            for item_id, payload in data.items():
                if not isinstance(payload, dict):
                    logger.debug(
                        "KvRedis upsert storing non-dict payload for id '%s' (%s)",
                        item_id,
                        type(payload).__name__,
                    )
                pipe.set(self._make_key(item_id), _encode_payload(payload))
            await pipe.execute()
        logger.debug("KvRedis upserted %d records in namespace '%s'", len(data), self.namespace)

    async def drop(self) -> None:
        pattern = f"{self._namespace_prefix}*"
        keys_to_delete: list[bytes | str] = []
        cursor = 0
        while True:
            cursor, batch = await self._redis.scan(cursor=cursor, match=pattern, count=2048)
            if batch:
                keys_to_delete.extend(batch)
            if cursor == 0:
                break
        if not keys_to_delete:
            logger.info("KvRedis drop called for namespace '%s' but no keys matched", self.namespace)
            return
        async with self._redis.pipeline(transaction=False) as pipe:
            for key in keys_to_delete:
                pipe.delete(key)
            await pipe.execute()
        logger.info(
            "KvRedis dropped %d keys for namespace '%s'",
            len(keys_to_delete),
            self.namespace,
        )

