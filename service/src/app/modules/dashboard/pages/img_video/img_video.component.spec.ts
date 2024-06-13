import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImgVideoComponent } from './img_video.component';

describe('ImgVideoComponent', () => {
  let component: ImgVideoComponent;
  let fixture: ComponentFixture<ImgVideoComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
    imports: [ImgVideoComponent],
}).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ImgVideoComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
