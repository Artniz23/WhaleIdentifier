import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { AlertController, ToastController } from '@ionic/angular';
import { WhaleStateService } from '../../services/whale-state.service';
import { IdentificationResult, WhaleMatch } from '../../models/whale.models';

interface ResultVM extends IdentificationResult {
  decision: 'pending' | 'approved' | 'rejected';
  expanded: boolean;
}

@Component({
  selector: 'app-results',
  templateUrl: './results.page.html',
  styleUrls: ['./results.page.scss']
})
export class ResultsPage implements OnInit {
  results: ResultVM[] = [];

  constructor(
    private state: WhaleStateService,
    private router: Router,
    private alertCtrl: AlertController,
    private toastCtrl: ToastController
  ) {}

  ngOnInit(): void {
    const raw = this.state.results;
    if (!raw.length) {
      this.router.navigate(['/home']);
      return;
    }
    this.results = raw.map(r => ({
      ...r,
      decision: 'pending',
      expanded: true
    }));
  }

  get pendingCount(): number { return this.results.filter(r => r.decision === 'pending').length; }
  get approvedCount(): number { return this.results.filter(r => r.decision === 'approved').length; }
  get rejectedCount(): number { return this.results.filter(r => r.decision === 'rejected').length; }
  get allDecided(): boolean { return this.pendingCount === 0; }

  topMatch(result: ResultVM): WhaleMatch | null {
    return result.matches?.[0] ?? null;
  }

  confidencePct(confidence: number): number {
    return Math.round((confidence <= 1 ? confidence * 100 : confidence));
  }

  confidenceColor(confidence: number): string {
    const pct = this.confidencePct(confidence);
    if (pct >= 75) return 'success';
    if (pct >= 45) return 'warning';
    return 'danger';
  }

  toggleExpand(result: ResultVM): void { result.expanded = !result.expanded; }

  approve(result: ResultVM): void {
    result.decision = 'approved';
    this.showToast(
      result.is_new
        ? '✅ Новый кит одобрен для добавления в каталог'
        : `✅ Совпадение с китом ${this.topMatch(result)?.whale_id} подтверждено`,
      'success'
    );
  }

  reject(result: ResultVM): void {
    result.decision = 'rejected';
    this.showToast('❌ Результат отклонён', 'medium');
  }

  resetDecision(result: ResultVM): void { result.decision = 'pending'; }

  async finishSession(): Promise<void> {
    const alert = await this.alertCtrl.create({
      header: 'Завершить сессию?',
      message: `Одобрено: ${this.approvedCount} | Отклонено: ${this.rejectedCount}. Вы можете начать новую обработку видео.`,
      buttons: [
        { text: 'Отмена', role: 'cancel' },
        {
          text: 'Завершить',
          role: 'confirm',
          handler: () => {
            this.state.reset();
            this.router.navigate(['/home']);
          }
        }
      ]
    });
    await alert.present();
  }

  async newSession(): Promise<void> {
    if (!this.allDecided) {
      const alert = await this.alertCtrl.create({
        header: 'Есть нерешённые результаты',
        message: `${this.pendingCount} результат(ов) ожидают вашего решения. Начать новую сессию?`,
        buttons: [
          { text: 'Остаться', role: 'cancel' },
          {
            text: 'Начать заново',
            role: 'confirm',
            handler: () => {
              this.state.reset();
              this.router.navigate(['/home']);
            }
          }
        ]
      });
      await alert.present();
    } else {
      this.state.reset();
      this.router.navigate(['/home']);
    }
  }

  private async showToast(msg: string, color = 'primary'): Promise<void> {
    const toast = await this.toastCtrl.create({
      message: msg, duration: 3500, color, position: 'bottom'
    });
    await toast.present();
  }
}

