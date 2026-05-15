import { Component, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { Router } from '@angular/router';
import { AlertController, ToastController } from '@ionic/angular';
import { Subscription } from 'rxjs';
import { WhaleApiService } from '../../services/whale-api.service';
import { WhaleStateService } from '../../services/whale-state.service';

type PageStage = 'idle' | 'uploading' | 'processing' | 'done' | 'error';

@Component({
  selector: 'app-home',
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss']
})
export class HomePage implements OnDestroy {
  @ViewChild('fileInput') fileInputRef!: ElementRef<HTMLInputElement>;

  stage: PageStage = 'idle';
  uploadProgress = 0;
  processingProgress = 0;
  processingStage = '';
  processingMessage = '';
  isDragOver = false;
  errorMessage = '';

  private subs = new Subscription();

  constructor(
    private api: WhaleApiService,
    private state: WhaleStateService,
    private router: Router,
    private alertCtrl: AlertController,
    private toastCtrl: ToastController
  ) {}

  get selectedVideo(): File | null { return this.state.selectedVideo; }
  get videoPreviewUrl(): string | null { return this.state.videoPreviewUrl; }
  get videoName(): string { return this.selectedVideo?.name ?? ''; }
  get videoSize(): string {
    const bytes = this.selectedVideo?.size ?? 0;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }

  // ── Drag & Drop ──────────────────────────────────────────────────────────
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = true;
  }

  onDragLeave(): void { this.isDragOver = false; }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = false;
    const file = event.dataTransfer?.files?.[0];
    if (file) this.selectFile(file);
  }

  openFileDialog(): void { this.fileInputRef.nativeElement.click(); }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) this.selectFile(file);
    input.value = '';
  }

  selectFile(file: File): void {
    if (!file.type.startsWith('video/')) {
      this.showToast('Пожалуйста, выберите видеофайл', 'warning');
      return;
    }
    this.state.reset();
    this.state.setVideo(file);
    this.stage = 'idle';
    this.errorMessage = '';
  }

  clearVideo(): void {
    this.state.reset();
    this.stage = 'idle';
    this.errorMessage = '';
  }

  // ── Upload & Process ─────────────────────────────────────────────────────
  startProcessing(): void {
    if (!this.selectedVideo) return;
    this.stage = 'uploading';
    this.uploadProgress = 0;

    const sub = this.api.uploadVideo(this.selectedVideo).subscribe({
      next: event => {
        if (event.type === 'progress') {
          this.uploadProgress = event.value;
        } else if (event.type === 'done') {
          console.log('event', event);
          this.state.setJobId(event.jobId);
          this.startPolling(event.jobId);
        }
      },
      error: err => {
        this.stage = 'error';
        this.errorMessage = err?.error?.detail ?? 'Ошибка загрузки видео';
      }
    });
    this.subs.add(sub);
  }

  private startPolling(jobId: string): void {
    this.stage = 'processing';
    this.processingProgress = 0;
    this.processingMessage = 'Инициализация…';

    const sub = this.api.pollJobStatus(jobId).subscribe({
      next: status => {
        this.processingProgress = status.progress ?? 0;
        this.processingMessage = status.message ?? this.stageLabel(status.stage ?? '');
        this.processingStage = status.stage ?? '';

        if (status.status === 'completed') {
          this.fetchFrames(jobId);
        } else if (status.status === 'failed') {
          this.stage = 'error';
          this.errorMessage = status.message ?? 'Ошибка обработки видео';
        }
      },
      error: () => {
        this.stage = 'error';
        this.errorMessage = 'Потеряно соединение с сервером';
      }
    });
    this.subs.add(sub);
  }

  private fetchFrames(jobId: string): void {
    const sub = this.api.getTracks(jobId).subscribe({
      next: tracks => {
        this.state.setTracks(tracks);
        this.stage = 'done';
        this.router.navigate(['/frames']);
      },
      error: () => {
        this.stage = 'error';
        this.errorMessage = 'Не удалось получить кадры с сервера';
      }
    });
    this.subs.add(sub);
  }

  private stageLabel(stage: string): string {
    const map: Record<string, string> = {
      detection: 'Детекция китов…',
      extraction: 'Выбор лучших кадров…',
      scoring: 'Оценка качества кадров…',
      upload: 'Загрузка файла…'
    };
    return map[stage] ?? 'Обработка видео…';
  }

  retry(): void {
    this.stage = 'idle';
    this.errorMessage = '';
  }

  private async showToast(msg: string, color = 'primary'): Promise<void> {
    const toast = await this.toastCtrl.create({
      message: msg, duration: 3000, color, position: 'bottom'
    });
    await toast.present();
  }

  ngOnDestroy(): void { this.subs.unsubscribe(); }
}


