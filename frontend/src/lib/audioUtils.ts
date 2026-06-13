export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      if (typeof reader.result === 'string') {
        // Remove the data URI prefix (e.g., "data:audio/mp3;base64,")
        const base64 = reader.result.split(',')[1];
        if (base64) {
          resolve(base64);
        } else {
          reject(new Error('Failed to extract base64 data from file'));
        }
      } else {
        reject(new Error('FileReader result is not a string'));
      }
    };

    reader.onerror = () => {
      reject(new Error(`FileReader error: ${reader.error?.message ?? 'Unknown error'}`));
    };

    reader.readAsDataURL(file);
  });
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function getAudioFormat(file: File): string {
  const ext = file.name.split('.').pop()?.toLowerCase();
  if (ext === 'wav') return 'wav';
  return 'mp3';
}
