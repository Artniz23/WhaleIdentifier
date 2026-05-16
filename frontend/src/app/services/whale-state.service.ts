import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { Track, IdentificationGroupResult } from '../models/whale.models';

@Injectable({ providedIn: 'root' })
export class WhaleStateService {
  private _jobId = new BehaviorSubject<string | null>(null);
  private _frames = new BehaviorSubject<Track[]>([]);
  private _results = new BehaviorSubject<IdentificationGroupResult[]>([]);
  private _selectedVideo = new BehaviorSubject<File | null>(null);
  private _videoPreviewUrl = new BehaviorSubject<string | null>(null);

  readonly jobId$ = this._jobId.asObservable();
  readonly frames$ = this._frames.asObservable();
  readonly results$ = this._results.asObservable();
  readonly selectedVideo$ = this._selectedVideo.asObservable();
  readonly videoPreviewUrl$ = this._videoPreviewUrl.asObservable();

  get jobId(): string | null { return this._jobId.value; }
  get frames(): Track[] { return this._frames.value; }
  get results(): IdentificationGroupResult[] { return this._results.value; }
  get selectedVideo(): File | null { return this._selectedVideo.value; }
  get videoPreviewUrl(): string | null { return this._videoPreviewUrl.value; }

  setJobId(id: string): void { this._jobId.next(id); }
  setTracks(tracks: Track[]): void { this._frames.next(tracks); }
  setResults(results: IdentificationGroupResult[]): void { this._results.next(results); }

  setVideo(file: File): void {
    // Revoke previous object URL to avoid memory leaks
    if (this._videoPreviewUrl.value) {
      URL.revokeObjectURL(this._videoPreviewUrl.value);
    }
    this._selectedVideo.next(file);
    this._videoPreviewUrl.next(URL.createObjectURL(file));
  }

  reset(): void {
    if (this._videoPreviewUrl.value) {
      URL.revokeObjectURL(this._videoPreviewUrl.value);
    }
    this._jobId.next(null);
    this._frames.next([]);
    this._results.next([]);
    this._selectedVideo.next(null);
    this._videoPreviewUrl.next(null);
  }
}

