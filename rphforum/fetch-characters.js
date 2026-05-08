import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import sharp from 'sharp';
import HuggingFaceDatasetService from './HuggingFaceDatasetService.js';
import dotenv from 'dotenv';

// 加载环境变量
dotenv.config();

// ========== 配置区域 ==========
// 从环境变量读取配置
const HF_CONFIG = {
  enabled: process.env.RPH_HF_ENABLED !== 'false',
  repo: process.env.RPH_HF_REPO || 'dasodefa/role-play-data',
  token: process.env.RPH_HF_TOKEN || '',
  proxyUrl: process.env.RPH_HF_PROXY_URL || null
};

const BATCH_SIZE = parseInt(process.env.RPH_HF_BATCH_SIZE) || 2; // 默认一批 2 个文件
// ================================

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const API_BASE_URL = 'https://rphforum.zeabur.app/api';
const CHARACTERS_DIR = path.join(__dirname, 'characters');
const DATA_DIR = path.join(CHARACTERS_DIR, 'data');
const THUMBNAILS_DIR = path.join(CHARACTERS_DIR, 'thumbnails');
const REQUEST_DELAY = 500;
const THUMBNAIL_SIZE = 200;
const THUMBNAIL_QUALITY = 80;

const hfService = new HuggingFaceDatasetService(HF_CONFIG);

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchCharacterList() {
  console.log('正在获取角色列表...');
  const response = await fetch(`${API_BASE_URL}/cards`);
  if (!response.ok) {
    throw new Error(`获取角色列表失败: ${response.status} ${response.statusText}`);
  }
  const characters = await response.json();
  console.log(`成功获取 ${characters.length} 个角色`);
  return characters;
}

function sortByHotness(characters) {
  return [...characters].sort((a, b) => {
    const hotnessA = (a.views_count || 0) + (a.downloads_count || 0) + (a.comment_count || 0);
    const hotnessB = (b.views_count || 0) + (b.downloads_count || 0) + (b.comment_count || 0);
    return hotnessB - hotnessA;
  });
}

async function fetchCharacterDetail(id) {
  console.log(`正在获取角色详情: ${id}`);
  const response = await fetch(`${API_BASE_URL}/cards/${id}`);
  if (!response.ok) {
    throw new Error(`获取角色详情失败 ${id}: ${response.status} ${response.statusText}`);
  }
  return await response.json();
}

function parseBase64Image(base64Data) {
  const matches = base64Data.match(/^data:([^;]+);base64,(.+)$/);
  if (!matches) {
    throw new Error('无效的 base64 图片格式');
  }
  const mimeType = matches[1];
  const data = matches[2];
  const extension = mimeType.split('/')[1] || 'png';
  return { extension, buffer: Buffer.from(data, 'base64') };
}

class FileUploadQueue {
  constructor(service, batchSize) {
    this.service = service;
    this.batchSize = batchSize;
    this.queue = [];
  }

  async add(file) {
    this.queue.push(file);
    
    if (this.queue.length >= this.batchSize) {
      await this.flush();
    }
  }

  async flush() {
    if (this.queue.length === 0) return;
    
    console.log(`\n📤 开始批量上传 ${this.queue.length} 个文件...`);
    
    try {
      await this.service.uploadFiles(this.queue, {
        message: `Batch upload of ${this.queue.length} files`
      });
      console.log(`✅ 成功批量上传 ${this.queue.length} 个文件\n`);
    } catch (error) {
      console.error(`❌ 批量上传失败:`, error);
    }
    
    this.queue = [];
  }

  size() {
    return this.queue.length;
  }
}

async function saveCharacterData(character, uploadQueue = null) {
  if (!character.avatar_url) {
    console.log(`  角色 ${character.id} 没有 avatar`);
    return;
  }

  try {
    const { buffer: avatarBuffer } = parseBase64Image(character.avatar_url);
    
    const pngAvatarBuffer = await sharp(avatarBuffer).png().toBuffer();
    
    const thumbnailBuffer = await sharp(avatarBuffer)
      .resize(THUMBNAIL_SIZE, THUMBNAIL_SIZE, { fit: 'cover' })
      .jpeg({ quality: THUMBNAIL_QUALITY })
      .toBuffer();

    if (hfService.isEnabled() && uploadQueue) {
      const dataPath = `characters/data/${character.id}.png`;
      const thumbnailPath = `characters/thumbnails/${character.id}.jpg`;
      
      // 添加到队列中
      await uploadQueue.add({
        path: dataPath,
        content: pngAvatarBuffer
      });
      
      await uploadQueue.add({
        path: thumbnailPath,
        content: thumbnailBuffer
      });
      
      console.log(`  ✅ 角色 ${character.id} 已加入上传队列`);
    } else if (hfService.isEnabled()) {
      await hfService.initialize();
      
      const dataPath = `characters/data/${character.id}.png`;
      const thumbnailPath = `characters/thumbnails/${character.id}.jpg`;
      
      await hfService.uploadFiles([
        {
          path: dataPath,
          content: pngAvatarBuffer
        },
        {
          path: thumbnailPath,
          content: thumbnailBuffer
        }
      ], {
        message: `Upload character ${character.id} data and thumbnail`
      });
      
      console.log(`  ✅ 角色数据已上传: ${dataPath}`);
      console.log(`  ✅ 缩略图已上传: ${thumbnailPath}`);
    } else {
      if (!fs.existsSync(DATA_DIR)) {
        fs.mkdirSync(DATA_DIR, { recursive: true });
      }
      if (!fs.existsSync(THUMBNAILS_DIR)) {
        fs.mkdirSync(THUMBNAILS_DIR, { recursive: true });
      }

      const dataPath = path.join(DATA_DIR, `${character.id}.png`);
      fs.writeFileSync(dataPath, pngAvatarBuffer);
      console.log(`  角色数据已保存: ${dataPath}`);

      const thumbnailPath = path.join(THUMBNAILS_DIR, `${character.id}.jpg`);
      fs.writeFileSync(thumbnailPath, thumbnailBuffer);
      console.log(`  缩略图已保存: ${thumbnailPath}`);
    }

  } catch (error) {
    console.error(`  保存角色数据失败 ${character.id}:`, error.message);
  }
}

function parseCountArg() {
  const arg = process.argv[2];
  if (arg === undefined) {
    return 1;
  }
  const count = parseInt(arg, 10);
  if (isNaN(count) || count < 0) {
    console.warn('无效的数量参数，使用默认值 1');
    return 1;
  }
  return count;
}

async function main() {
  console.log('========================================');
  console.log('        角色抓取和处理程序');
  console.log('========================================');
  console.log(`批量上传大小: ${BATCH_SIZE} 个文件`);
  console.log('========================================\n');
  
  try {
    const count = parseCountArg();
    console.log(`将处理 ${count === 0 ? '全部' : count} 个角色\n`);

    let characters = await fetchCharacterList();
    characters = sortByHotness(characters);

    const charactersToProcess = count === 0 ? characters : characters.slice(0, count);
    console.log(`\n开始处理 ${charactersToProcess.length} 个角色...\n`);

    // 创建上传队列
    let uploadQueue = null;
    if (hfService.isEnabled()) {
      await hfService.initialize();
      uploadQueue = new FileUploadQueue(hfService, BATCH_SIZE);
    }

    for (let i = 0; i < charactersToProcess.length; i++) {
      const char = charactersToProcess[i];
      console.log(`[${i + 1}/${charactersToProcess.length}] 处理角色: ${char.name} (${char.id})`);

      try {
        const detail = await fetchCharacterDetail(char.id);
        await saveCharacterData(detail, uploadQueue);
      } catch (error) {
        console.error(`  处理角色失败 ${char.id}:`, error.message);
      }

      if (i < charactersToProcess.length - 1) {
        await delay(REQUEST_DELAY);
      }
    }

    // 上传队列中剩余的文件
    if (uploadQueue && uploadQueue.size() > 0) {
      console.log('\n========================================');
      console.log('  处理完成，上传剩余的文件...');
      console.log('========================================');
      await uploadQueue.flush();
    }

    console.log('\n========================================');
    console.log('              ✅ 处理完成!');
    console.log('========================================');
  } catch (error) {
    console.error('错误:', error.message);
    process.exit(1);
  }
}

main();
