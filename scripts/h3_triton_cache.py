"""Keep Triton cache temporary files beside their final path on the platform."""
import os,tempfile
from triton.runtime.cache import FileCacheManager
class SiblingCacheManager(FileCacheManager):
 def put(self,data,filename,binary=True):
  payload=data if isinstance(data,bytes) else str(data).encode()
  path=self._make_path(filename)
  fd,temp=tempfile.mkstemp(prefix=".h3-",suffix=".tmp",dir=os.path.dirname(path))
  try:
   with os.fdopen(fd,"wb") as f:
    f.write(payload);f.flush();os.fsync(f.fileno())
   os.replace(temp,path)
  finally:
   if os.path.exists(temp):os.unlink(temp)
  return path
