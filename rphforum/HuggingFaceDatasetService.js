import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import { uploadFiles as hfUploadFiles, deleteFile, listFiles } from '@huggingface/hub';

const HF_API_BASE = 'https://huggingface.co';

class HuggingFaceDatasetService {
  constructor(config) {
    this.repo = config.repo || '';
    this.token = config.token || '';
    this.initialEnabled = config.enabled && !!this.repo && !!this.token;
    this.enabled = this.initialEnabled;
    this.proxyUrl = config.proxyUrl || process.env.HF_PROXY_URL || null;
    this.cache = new Map();
    this.cacheDir = config.cacheDir || null;
    this.cacheTTL = config.cacheTTL || 3600000;
    this.initialized = false;
    this.lastFailedAt = null;
    this.minRetryInterval = 60000;
  }

  isEnabled() {
    return this.initialEnabled;
  }

  async initialize() {
    if (!this.initialEnabled) {
      return false;
    }

    if (this.initialized) {
      return true;
    }

    try {
      const exists = await this.repoExists();
      if (!exists) {
        console.warn(`[HuggingFace] Dataset repo ${this.repo} not found or no access`);
        this.lastFailedAt = Date.now();
        return false;
      }

      if (this.cacheDir) {
        try {
          await fs.access(this.cacheDir);
        } catch {
          await fs.mkdir(this.cacheDir, { recursive: true });
        }
      }

      this.initialized = true;
      this.lastFailedAt = null;
      console.log(`[HuggingFace] Dataset service initialized for ${this.repo}`);
      return true;
    } catch (error) {
      console.error('[HuggingFace] Failed to initialize:', error);
      this.lastFailedAt = Date.now();
      return false;
    }
  }

  async repoExists() {
    try {
      const response = await fetch(`${HF_API_BASE}/api/datasets/${this.repo}`, {
        headers: {
          'Authorization': `Bearer ${this.token}`
        }
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async uploadFile(filePath, content, options = {}) {
    if (!this.enabled) {
      throw new Error('HuggingFace Dataset service is not enabled');
    }

    try {
      const repoPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
      const commitMessage = options.message || `Upload ${repoPath}`;

      let blob;
      if (Buffer.isBuffer(content)) {
        blob = new Blob([content]);
      } else if (typeof content === 'string') {
        blob = new Blob([content]);
      } else {
        blob = new Blob([JSON.stringify(content, null, 2)]);
      }

      console.log(`[HuggingFace] Uploading file: ${repoPath} (${blob.size} bytes)`);

      await hfUploadFiles({
        repo: { type: 'dataset', name: this.repo },
        credentials: { accessToken: this.token },
        files: [{
          path: repoPath,
          content: blob
        }],
        title: commitMessage
      });

      const cacheKey = this.getCacheKey(repoPath);
      this.cache.delete(cacheKey);

      console.log(`[HuggingFace] ✅ Uploaded file: ${repoPath}`);
      return { success: true, path: repoPath };
    } catch (error) {
      console.error(`[HuggingFace] Failed to upload file ${filePath}:`, error);
      throw error;
    }
  }

  async uploadFiles(fileList, options = {}) {
    if (!this.enabled) {
      throw new Error('HuggingFace Dataset service is not enabled');
    }

    try {
      const commitMessage = options.message || 'Upload multiple files';
      
      const files = fileList.map(fileItem => {
        const repoPath = fileItem.path.startsWith('/') ? fileItem.path.slice(1) : fileItem.path;
        
        let blob;
        if (Buffer.isBuffer(fileItem.content)) {
          blob = new Blob([fileItem.content]);
        } else if (typeof fileItem.content === 'string') {
          blob = new Blob([fileItem.content]);
        } else {
          blob = new Blob([JSON.stringify(fileItem.content, null, 2)]);
        }
        
        console.log(`[HuggingFace] Preparing file: ${repoPath} (${blob.size} bytes)`);
        
        return {
          path: repoPath,
          content: blob
        };
      });

      console.log(`[HuggingFace] Uploading ${files.length} files...`);

      await hfUploadFiles({
        repo: { type: 'dataset', name: this.repo },
        credentials: { accessToken: this.token },
        files: files,
        title: commitMessage
      });

      // Clear cache for all uploaded files
      files.forEach(file => {
        const cacheKey = this.getCacheKey(file.path);
        this.cache.delete(cacheKey);
      });

      console.log(`[HuggingFace] ✅ Uploaded ${files.length} files`);
      return { success: true, paths: files.map(f => f.path) };
    } catch (error) {
      console.error(`[HuggingFace] Failed to upload files:`, error);
      throw error;
    }
  }

  async downloadFile(filePath, options = {}) {
    if (!this.enabled) {
      throw new Error('HuggingFace Dataset service is not enabled');
    }

    const repoPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
    const cacheKey = this.getCacheKey(repoPath);

    if (!options.skipCache) {
      const cached = this.cache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
        return cached.data;
      }
    }

    try {
      const branch = options.branch || 'main';
      const response = await fetch(`${HF_API_BASE}/datasets/${this.repo}/resolve/${branch}/${repoPath}`, {
        headers: {
          'Authorization': `Bearer ${this.token}`
        }
      });

      if (!response.ok) {
        if (response.status === 404) {
          return null;
        }
        throw new Error(`Failed to download file: ${response.status}`);
      }

      const contentType = response.headers.get('content-type') || '';
      let data;

      if (contentType.includes('application/json') || repoPath.endsWith('.json')) {
        data = await response.json();
      } else {
        const arrayBuffer = await response.arrayBuffer();
        data = Buffer.from(arrayBuffer);
      }

      this.cache.set(cacheKey, {
        data,
        timestamp: Date.now()
      });

      console.log(`[HuggingFace] Downloaded file: ${repoPath}`);
      return data;
    } catch (error) {
      console.error(`[HuggingFace] Failed to download file ${filePath}:`, error);
      throw error;
    }
  }

  async fileExists(filePath, options = {}) {
    if (!this.enabled) {
      return false;
    }

    try {
      const repoPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
      const branch = options.branch || 'main';

      const response = await fetch(`${HF_API_BASE}/datasets/${this.repo}/resolve/${branch}/${repoPath}`, {
        method: 'HEAD',
        headers: {
          'Authorization': `Bearer ${this.token}`
        }
      });

      return response.ok;
    } catch {
      return false;
    }
  }

  getCacheKey(repoPath) {
    return crypto.createHash('md5').update(`${this.repo}:${repoPath}`).digest('hex');
  }
}

export default HuggingFaceDatasetService;
