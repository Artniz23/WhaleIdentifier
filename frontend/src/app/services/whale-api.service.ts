import { Injectable } from '@angular/core';
import { HttpClient, HttpEventType, HttpRequest } from '@angular/common/http';
import { Observable, interval, throwError } from 'rxjs';
import { switchMap, takeWhile, filter, map, catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import {
  UploadResponse,
  JobStatus,
  Track,
  IdentifyResponse,
  IdentifySelection
} from '../models/whale.models';

@Injectable({ providedIn: 'root' })
export class WhaleApiService {
  private readonly base = environment.apiUrl;

  constructor(private http: HttpClient) {}

  /**
   * Upload video file to the backend.
   * Returns an observable that emits upload progress (0-100) and then the job_id.
   */
  uploadVideo(file: File): Observable<{ type: 'progress'; value: number } | { type: 'done'; jobId: string }> {
    const formData = new FormData();
    formData.append('video', file);

    const req = new HttpRequest('POST', `${this.base}/upload`, formData, {
      reportProgress: true
    });

    return this.http.request(req).pipe(
      map(event => {
        if (event.type === HttpEventType.UploadProgress) {
          const progress = event.total
            ? Math.round((100 * event.loaded) / event.total)
            : 0;
          return { type: 'progress' as const, value: progress };
        } else if (event.type === HttpEventType.Response) {
          const body = event.body as UploadResponse;
          return { type: 'done' as const, jobId: body.job_id };
        }
        return null;
      }),
      filter((v): v is { type: 'progress'; value: number } | { type: 'done'; jobId: string } => v !== null),
      catchError(err => throwError(() => err))
    );
  }

  /** Poll job status every 2 seconds until completed or failed */
  pollJobStatus(jobId: string): Observable<JobStatus> {
    return interval(2000).pipe(
      switchMap(() => this.http.get<JobStatus>(`${this.base}/job/${jobId}/status`)),
      takeWhile(s => s.status === 'pending' || s.status === 'processing', true)
    );
  }

  /** Get best frames selected by the AI */
  getTracks(jobId: string): Observable<Track[]> {
    return this.http.get<{ tracks: Track[] }>(`${this.base}/job/${jobId}/tracks`).pipe(
      map(r => r.tracks)
    );
  }

  /** Send selected frame IDs grouped by track for whale identification */
  identifyWhales(jobId: string, selections: IdentifySelection[]): Observable<IdentifyResponse> {
    return this.http.post<IdentifyResponse>(`${this.base}/identify`, {
      job_id: jobId,
      selections
    });
  }

  /** Poll identification job status */
  pollIdentifyStatus(jobId: string): Observable<JobStatus> {
    return interval(2000).pipe(
      switchMap(() => this.http.get<JobStatus>(`${this.base}/identify/${jobId}/status`)),
      takeWhile(s => s.status === 'pending' || s.status === 'processing', true)
    );
  }
}

