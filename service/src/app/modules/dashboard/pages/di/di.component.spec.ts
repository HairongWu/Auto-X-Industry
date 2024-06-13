import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DIComponent } from './di.component';

describe('NftComponent', () => {
  let component: DIComponent;
  let fixture: ComponentFixture<DIComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
    imports: [DIComponent],
}).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DIComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
