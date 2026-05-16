import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router } from '@angular/router';
import { AlertController, ToastController } from '@ionic/angular';
import { Subscription } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { WhaleApiService } from '../../services/whale-api.service';
import { WhaleStateService } from '../../services/whale-state.service';
import { Track, Frame } from '../../models/whale.models';

interface SelectableTrackFrame extends Frame {
  trackId: string;
  selected: boolean;
}

interface SelectableTrack extends Track {
  frames: SelectableTrackFrame[];
}

@Component({
  selector: 'app-frames',
  templateUrl: './frames.page.html',
  styleUrls: ['./frames.page.scss']
})
export class FramesPage implements OnInit, OnDestroy {
  tracks: SelectableTrack[] = [];
  isIdentifying = false;
  identifyProgress = 0;
  previewFrame: SelectableTrackFrame | null = null;

  private subs = new Subscription();

  constructor(
    private state: WhaleStateService,
    private api: WhaleApiService,
    private router: Router,
    private alertCtrl: AlertController,
    private toastCtrl: ToastController
  ) {}

  ngOnInit(): void {
    const tracks = this.state.frames;
    if (!tracks.length) {
      this.router.navigate(['/home']);
      return;
    }
    // All frames selected by default
    this.tracks = tracks.map(track => ({
      ...track,
      frames: track.frames.map(frame => ({
        ...frame,
        trackId: track.track_id,
        selected: true
      }))
    }));
  }

  get allFrames(): SelectableTrackFrame[] {
    return this.tracks.flatMap(track => track.frames);
  }

  get selectedFrames(): SelectableTrackFrame[] {
    return this.allFrames.filter(f => f.selected);
  }

  get selectedCount(): number { return this.selectedFrames.length; }
  get totalCount(): number { return this.allFrames.length; }
  get allSelected(): boolean {
    return this.allFrames.length > 0 && this.allFrames.every(f => f.selected);
  }
  get identifyProgressValue(): number {
    return Math.max(0, Math.min(100, Math.round(this.identifyProgress)));
  }
  get hasIdentifyProgress(): boolean { return this.identifyProgressValue > 0; }

  toggleFrame(frame: SelectableTrackFrame): void { frame.selected = !frame.selected; }

  toggleAll(): void {
    const selectAll = !this.allSelected;
    this.allFrames.forEach(f => f.selected = selectAll);
  }

  openPreview(frame: SelectableTrackFrame): void { this.previewFrame = frame; }
  closePreview(): void { this.previewFrame = null; }

  scorePercent(score: number): number { return Math.round(score * 100); }
  scoreColor(score: number): string {
    if (score >= 0.8) return 'success';
    if (score >= 0.5) return 'warning';
    return 'danger';
  }

  formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  async confirmIdentify(): Promise<void> {
    if (this.selectedCount === 0) {
      this.showToast('Выберите хотя бы один кадр', 'warning');
      return;
    }

    const alert = await this.alertCtrl.create({
      header: 'Начать идентификацию?',
      message: `Выбрано ${this.selectedCount} из ${this.totalCount} кадров. ИИ сравнит их с каталогом китов.`,
      buttons: [
        { text: 'Отмена', role: 'cancel' },
        {
          text: 'Идентифицировать',
          role: 'confirm',
          handler: () => this.startIdentification()
        }
      ]
    });
    await alert.present();
  }

  private startIdentification(): void {
    const jobId = this.state.jobId;
    if (!jobId) return;

    this.isIdentifying = true;
    this.identifyProgress = 0;
    const selectedIds = this.selectedFrames.map(f => f.id);

    const sub = this.api.identifyWhales(jobId, selectedIds).pipe(
      switchMap(response => {
        if (!response?.job_id) {
          throw new Error('Сервер не вернул ID задачи идентификации');
        }
        return this.api.pollIdentifyStatus(response.job_id);
      })
    ).subscribe({
      next: status => {
        if (typeof status.progress === 'number' && Number.isFinite(status.progress)) {
          this.identifyProgress = status.progress;
        }

        if (status.status === 'completed') {
          const results = status.results ?? [];
          this.state.setResults(results);
          this.isIdentifying = false;
          this.router.navigate(['/results']);
          return;
        }

        if (status.status === 'failed') {
          this.isIdentifying = false;
          this.showToast(status.message ?? 'Идентификация завершилась с ошибкой.', 'danger');
        }
      },
      error: err => {
        this.isIdentifying = false;
        this.showToast(
          err?.error?.detail ?? err?.message ?? 'Ошибка идентификации. Попробуйте ещё раз.',
          'danger'
        );
      }
    });
    this.subs.add(sub);
  }

  goBack(): void { this.router.navigate(['/home']); }

  private async showToast(msg: string, color = 'primary'): Promise<void> {
    const toast = await this.toastCtrl.create({
      message: msg, duration: 3000, color, position: 'bottom'
    });
    await toast.present();
  }

  ngOnDestroy(): void { this.subs.unsubscribe(); }
}

